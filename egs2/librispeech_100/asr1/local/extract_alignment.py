#!/usr/bin/env python3

""" Inference class definition for Transducer models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from packaging.version import parse as V
from typeguard import check_argument_types, check_return_type

from espnet2.asr_transducer.beam_search_transducer import (
    BeamSearchTransducer,
    Hypothesis,
)
from espnet2.asr_transducer.frontend.online_audio_processor import OnlineAudioProcessor
from espnet2.asr_transducer.utils import TooShortUttError
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.tasks.asr_transducer_otg import ASRTransducerTask
from espnet2.tasks.lm import LMTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.asr_transducer.utils import get_transducer_task_io

from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from espnet.utils.cli_utils import get_commandline_args


class Speech2Text:
    """Speech2Text class for Transducer models.

    Args:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
        beam_search_config: Beam search config path.
        lm_train_config: Language Model training config path.
        lm_file: Language Model config path.
        token_type: Type of token units.
        bpemodel: BPE model path.
        device: Device to use for inference.
        beam_size: Size of beam during search.
        dtype: Data type.
        lm_weight: Language model weight.
        quantize_asr_model: Whether to apply dynamic quantization to ASR model.
        quantize_modules: List of module names to apply dynamic quantization on.
        quantize_dtype: Dynamic quantization data type.
        nbest: Number of final hypothesis.
        streaming: Whether to perform chunk-by-chunk inference.
        decoding_window: Size of the decoding window (in milliseconds).
        left_context: Number of previous frames the attention module can see
                      in current chunk (used by Conformer and Branchformer block).

    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
    ) -> None:
        """Construct a Speech2Text object."""
        super().__init__()

        assert check_argument_types()

        beam_search_config = None
        lm_train_config = None
        lm_file = None
        token_type = None
        bpemodel = None
        device = "cpu"
        beam_size = 5
        dtype = "float32"
        lm_weight = 1.0
        quantize_asr_model = False
        quantize_modules = None
        quantize_dtype = "qint8"
        nbest = 1
        streaming = False
        decoding_window: int = 640,
        left_context = 32

        asr_model, asr_train_args = ASRTransducerTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )

        if quantize_asr_model:
            if quantize_modules is not None:
                if not all([q in ["LSTM", "Linear"] for q in quantize_modules]):
                    raise ValueError(
                        "Only 'Linear' and 'LSTM' modules are currently supported"
                        " by PyTorch and in --quantize_modules"
                    )

                q_config = set([getattr(torch.nn, q) for q in quantize_modules])
            else:
                q_config = {torch.nn.Linear}

            if quantize_dtype == "float16" and (V(torch.__version__) < V("1.5.0")):
                raise ValueError(
                    "float16 dtype for dynamic quantization is not supported with torch"
                    " version < 1.5.0. Switching to qint8 dtype instead."
                )
            q_dtype = getattr(torch, quantize_dtype)

            asr_model = torch.quantization.quantize_dynamic(
                asr_model, q_config, dtype=q_dtype
            ).eval()
        else:
            asr_model.to(dtype=getattr(torch, dtype)).eval()

        if hasattr(asr_model.decoder, "rescale_every") and (
            asr_model.decoder.rescale_every > 0
        ):
            rescale_every = asr_model.decoder.rescale_every

            with torch.no_grad():
                for block_id, block in enumerate(asr_model.decoder.rwkv_blocks):
                    block.att.proj_output.weight.div_(
                        2 ** int(block_id // rescale_every)
                    )
                    block.ffn.proj_value.weight.div_(
                        2 ** int(block_id // rescale_every)
                    )

            asr_model.decoder.rescaled_layers = True

        if lm_train_config is not None:
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config, lm_file, device
            )
            lm_scorer = lm.lm
        else:
            lm_scorer = None

        # 4. Build BeamSearch object
        if beam_search_config is None:
            beam_search_config = {}

        beam_search = BeamSearchTransducer(
            asr_model.decoder,
            asr_model.joint_network,
            beam_size,
            lm=lm_scorer,
            lm_weight=lm_weight,
            nbest=nbest,
            **beam_search_config,
        )

        token_list = asr_model.token_list

        if token_type is None:
            token_type = asr_train_args.token_type

        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type == "bpe":
            if bpemodel is not None:
                tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
            else:
                tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)
        converter = TokenIDConverter(token_list=token_list)

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        self.converter = converter
        self.tokenizer = tokenizer

        self.beam_search = beam_search

        self.streaming = streaming and decoding_window >= 0
        self.asr_model.encoder.dynamic_chunk_training = False
        self.left_context = max(left_context, 0)

        if streaming:
            self.audio_processor = OnlineAudioProcessor(
                asr_model._extract_feats,
                asr_model.normalize,
                decoding_window,
                asr_model.encoder.embed.subsampling_factor,
                asr_train_args.frontend_conf,
                device,
            )

            self.reset_streaming_cache()

    def reset_streaming_cache(self) -> None:
        """Reset Speech2Text parameters."""

        self.asr_model.encoder.reset_cache(self.left_context, device=self.device)
        self.beam_search.reset_cache()
        self.audio_processor.reset_cache()

        self.num_processed_frames = torch.tensor([[0]], device=self.device)


    @torch.no_grad()
    def streaming_decode(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        is_final: bool = False,
    ) -> List[Hypothesis]:
        """Speech2Text streaming call.

        Args:
            speech: Chunk of speech data. (S)
            is_final: Whether speech corresponds to the final chunk of data.

        Returns:
            nbest_hypothesis: N-best hypothesis.

        """
        nbest_hyps = []

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        speech = speech.to(device=self.device)

        feats, feats_length = self.audio_processor.compute_features(
            speech.to(getattr(torch, self.dtype)), is_final
        )

        enc_out = self.asr_model.encoder.chunk_forward(
            feats,
            feats_length,
            self.num_processed_frames,
            left_context=self.left_context,
        )
        self.num_processed_frames += enc_out.size(1)

        nbest_hyps = self.beam_search(enc_out[0], is_final=is_final)

        if is_final:
            self.reset_streaming_cache()

        return nbest_hyps

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> List[Hypothesis]:
        """Speech2Text call.

        Args:
            speech: Speech data. (S)

        Returns:
            nbest_hypothesis: N-best hypothesis.

        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        speech = speech.unsqueeze(0).to(
            dtype=getattr(torch, self.dtype), device=self.device
        )
        lengths = speech.new_full(
            [1], dtype=torch.long, fill_value=speech.size(1), device=self.device
        )

        feats, feats_length = self.asr_model._extract_feats(speech, lengths)

        if self.asr_model.normalize is not None:
            feats, feats_length = self.asr_model.normalize(feats, feats_length)

        enc_out, _ = self.asr_model.encoder(feats, feats_length)
        

        nbest_hyps = self.beam_search(enc_out[0])

        return nbest_hyps

    def hypotheses_to_results(self, nbest_hyps: List[Hypothesis]) -> List[Any]:
        """Build partial or final results from the hypotheses.

        Args:
            nbest_hyps: N-best hypothesis.

        Returns:
            results: Results containing different representation for the hypothesis.

        """
        results = []

        for hyp in nbest_hyps:
            token_int = list(filter(lambda x: x != 0, hyp.yseq))

            token = self.converter.ids2tokens(token_int)

            if self.tokenizer is not None:
                text = self.tokenizer.tokens2text(token)
            else:
                text = None
            results.append((text, token, token_int, hyp))

            assert check_return_type(results)

        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> Speech2Text:
        """Build Speech2Text instance from the pretrained model.

        Args:
            model_tag: Model tag of the pretrained models.

        Return:
            : Speech2Text instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        return Speech2Text(**kwargs)

def extract(
    
)

def get_parser():
    """Get Transducer model inference parser."""

    parser = config_argparse.ArgumentParser(
        description="ASR Transducer Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--asr_train_config",
        type=str,
        help="ASR training configuration",
    )
    group.add_argument(
        "--asr_model_file",
        type=str,
        help="ASR model parameter file",
    )

    return parser

def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)

    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    kwargs.pop("config", None)

    espnet_model = Speech2Text(**kwargs) 


if __name__ == "__main__":
    main()
