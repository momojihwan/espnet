"""ESPnet2 ASR Transducer model."""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import ot
import torch
import torch.nn.functional as F

from packaging.version import parse as V
from typeguard import check_argument_types

import matplotlib.pyplot as plt
import numpy as np


from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.encoder.encoder import Encoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:

    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASROTTransducerModel(AbsESPnetModel):
    """ESPnet2ASRTransducerModel module definition.

    Args:
        vocab_size: Size of complete vocabulary (w/ SOS/EOS and blank included).
        token_list: List of tokens in vocabulary (minus reserved tokens).
        frontend: Frontend module.
        specaug: SpecAugment module.
        normalize: Normalization module.
        encoder: Encoder module.
        decoder: Decoder module.
        joint_network: Joint Network module.
        transducer_weight: Weight of the Transducer loss.
        use_k2_pruned_loss: Whether to use k2 pruned Transducer loss.
        k2_pruned_loss_args: Arguments of the k2 loss pruned Transducer loss.
        warmup_steps: Number of steps in warmup, used for pruned loss scaling.
        validation_nstep: Maximum number of symbol expansions at each time step
                          when reporting CER or/and WER using mAES.
        fastemit_lambda: FastEmit lambda value.
        auxiliary_ctc_weight: Weight of auxiliary CTC loss.
        auxiliary_ctc_dropout_rate: Dropout rate for auxiliary CTC loss inputs.
        auxiliary_lm_loss_weight: Weight of auxiliary LM loss.
        auxiliary_lm_loss_smoothing: Smoothing rate for LM loss' label smoothing.
        ignore_id: Initial padding ID.
        sym_space: Space symbol.
        sym_blank: Blank Symbol.
        report_cer: Whether to report Character Error Rate during validation.
        report_wer: Whether to report Word Error Rate during validation.
        extract_feats_in_collect_stats: Whether to use extract_feats stats collection.

    """

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: Encoder,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        teacher_model: AbsESPnetModel,
        transducer_weight: float = 1.0,
        kd_weight: float = 0.3,
        temp_tau: float = 1.0,
        use_k2_pruned_loss: bool = False,
        k2_pruned_loss_args: Dict = {},
        warmup_steps: int = 25000,
        inter_weight: float= 0.5,
        validation_nstep: int = 2,
        fastemit_lambda: float = 0.0,
        auxiliary_ctc_weight: float = 0.0,
        auxiliary_ctc_dropout_rate: float = 0.0,
        auxiliary_lm_loss_weight: float = 0.0,
        auxiliary_lm_loss_smoothing: float = 0.05,
        ignore_id: int = -1,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        report_cer: bool = False,
        report_wer: bool = False,
        extract_feats_in_collect_stats: bool = True,
    ) -> None:
        """Construct an ESPnetASRTransducerModel object."""
        super().__init__()

        assert check_argument_types()

        # The following labels ID are reserved:
        #    - 0: Blank symbol.
        #    - 1: Unknown symbol.
        #    - vocab_size - 1: SOS/EOS symbol.
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

        self.sym_space = sym_space
        self.sym_blank = sym_blank

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize

        self.encoder = encoder
        self.decoder = decoder
        self.joint_network = joint_network

        self.teacher_model = teacher_model
        self.slice_time = 512
        self.slice_label = 5
        self.lambda_pos = 0.1

        self.criterion_transducer = None
        self.error_calculator = None

        self.use_auxiliary_ctc = auxiliary_ctc_weight > 0
        self.use_auxiliary_lm_loss = auxiliary_lm_loss_weight > 0
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.gamma = 0.1
        self.epsilon = 1e-3  # Regularization parameter for Sinkhorn algorithm
        self.max_iter = 100  # Ma
        self.chunk_size = 100

        self.inter_weight = inter_weight

        if use_k2_pruned_loss:
            self.am_proj = torch.nn.Linear(
                encoder.output_size,
                vocab_size,
            )

            self.lm_proj = torch.nn.Linear(
                decoder.output_size,
                vocab_size,
            )

            self.warmup_steps = warmup_steps
            self.steps_num = -1

            self.k2_pruned_loss_args = k2_pruned_loss_args
            self.k2_loss_type = k2_pruned_loss_args.get("loss_type", "regular")

        self.use_k2_pruned_loss = use_k2_pruned_loss

        if self.use_auxiliary_ctc:
            self.ctc_lin = torch.nn.Linear(encoder.output_size, vocab_size)
            self.ctc_dropout_rate = auxiliary_ctc_dropout_rate

        if self.use_auxiliary_lm_loss:
            self.lm_lin = torch.nn.Linear(decoder.output_size, vocab_size)

            eps = auxiliary_lm_loss_smoothing / (vocab_size - 1)

            self.lm_loss_smooth_neg = eps
            self.lm_loss_smooth_pos = (1 - auxiliary_lm_loss_smoothing) + eps

        self.transducer_weight = transducer_weight
        self.kd_weight = kd_weight
        self.fastemit_lambda = fastemit_lambda
        self.temp_tau = temp_tau

        self.auxiliary_ctc_weight = auxiliary_ctc_weight
        self.auxiliary_lm_loss_weight = auxiliary_lm_loss_weight

        self.report_cer = report_cer
        self.report_wer = report_wer
        self.validation_nstep = validation_nstep

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward architecture and compute loss(es).

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)
            text: Label ID sequences. (B, L)
            text_lengths: Label ID sequences lengths. (B,)
            kwargs: Contains "utts_id".

        Return:
            loss: Main loss value.
            stats: Task statistics.
            weight: Task weights.

        """
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        batch_size = speech.shape[0]
        text = text[:, : text_lengths.max()]

        with torch.no_grad():
            # 1-1. Teacher Encoder
            teacher_encoder_out, teacher_encoder_out_lens = self.teacher_encode(speech, speech_lengths)

            # 2-1. Teacher Transducer-related I/O preparation
            teacher_decoder_in, target, teacher_t_len, teacher_u_len, = get_transducer_task_io(
                text,
                teacher_encoder_out_lens,
                ignore_id=self.teacher_model.ignore_id,
            )

            # 3-1. Decoder
            self.teacher_model.decoder.set_device(teacher_encoder_out.device)
            teacher_decoder_out = self.teacher_model.decoder(teacher_decoder_in)

            # 4-1. Joint Network
            teacher_joint_out = self.teacher_model.joint_network(
                teacher_encoder_out.unsqueeze(2), teacher_decoder_out.unsqueeze(1)
            )

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2. Transducer-related I/O preparation
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            text,
            encoder_out_lens,
            ignore_id=self.ignore_id,
        )

        # 3. Decoder
        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        # 4. Joint Network and RNNT loss computation
        if self.use_k2_pruned_loss:
            loss_trans = self._calc_k2_transducer_pruned_loss(
                encoder_out, decoder_out, text, t_len, u_len, **self.k2_pruned_loss_args
            )
        else:
            joint_out = self.joint_network(
                encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
            )

            # Calculate underlying transducer loss
            loss_trans = self._calc_transducer_loss(
                encoder_out,
                joint_out,
                target,
                t_len,
                u_len,
            )


        # 4-0. Calculate OT loss
        loss_kd = self._calc_ot_loss(
            encoder_out,
            teacher_encoder_out,
            decoder_out,
            teacher_decoder_out,
        )

        # 4-0. Calculate differences between teacher and student logits then, maximize the differences
        # loss_kd = self.optimal_transport_alignment(joint_out, teacher_joint_out)

        # 4-1. Calculate knowledge distillation loss
        # loss_kd = self._calc_kd_loss(
        #     joint_out,
        #     opt_logits
        # )
        # 5. Auxiliary losses
        loss_ctc, loss_lm = 0.0, 0.0

        if self.use_auxiliary_ctc:
            loss_ctc = self._calc_ctc_loss(
                encoder_out,
                target,
                t_len,
                u_len,
            )

        if self.use_auxiliary_lm_loss:
            loss_lm = self._calc_lm_loss(decoder_out, target)

        loss = (
            self.transducer_weight * loss_trans
            + self.auxiliary_ctc_weight * loss_ctc
            + self.auxiliary_lm_loss_weight * loss_lm
        )

        loss = self.kd_weight * loss_kd + (1 - self.kd_weight) * loss

        # 6. CER/WER computation.
        if not self.training and (self.report_cer or self.report_wer):
            if self.error_calculator is None:
                from espnet2.asr_transducer.error_calculator import ErrorCalculator

                if self.use_k2_pruned_loss and self.k2_loss_type == "modified":
                    self.validation_nstep = 1

                self.error_calculator = ErrorCalculator(
                    self.decoder,
                    self.joint_network,
                    self.token_list,
                    self.sym_space,
                    self.sym_blank,
                    nstep=self.validation_nstep,
                    report_cer=self.report_cer,
                    report_wer=self.report_wer,
                )

            cer_transducer, wer_transducer = self.error_calculator(
                encoder_out, target, t_len
            )
        else:
            cer_transducer, wer_transducer = None, None

        stats = dict(
            loss=loss.detach(),
            loss_transducer=loss_trans.detach(),
            loss_kd=loss_kd.detach(),
            loss_aux_ctc=loss_ctc.detach() if loss_ctc > 0.0 else None,
            loss_aux_lm=loss_lm.detach() if loss_lm > 0.0 else None,
            cer_transducer=cer_transducer,
            wer_transducer=wer_transducer,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Collect features sequences and features lengths sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)
            text: Label ID sequences. (B, L)
            text_lengths: Label ID sequences lengths. (B,)
            kwargs: Contains "utts_id".

        Return:
            {}: "feats": Features sequences. (B, T, D_feats),
                "feats_lengths": Features sequences lengths. (B,)

        """
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )

            feats, feats_lengths = speech, speech_lengths

        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder speech sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            encoder_out: Encoder outputs. (B, T, D_enc)
            encoder_out_lens: Encoder outputs lengths. (B,)

        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Forward encoder
        encoder_out, encoder_out_lens = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def teacher_encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder speech sequences.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            encoder_out: Encoder outputs. (B, T, D_enc)
            encoder_out_lens: Encoder outputs lengths. (B,)

        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self.teacher_model._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.teacher_model.specaug is not None and self.training:
                feats, feats_lengths = self.teacher_model.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.teacher_model.normalize is not None:
                feats, feats_lengths = self.teacher_model.normalize(feats, feats_lengths)

        # 4. Forward encoder
        encoder_out, encoder_out_lens = self.teacher_model.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def sinkhorn_knopp(self, cost_matrix):
        B, N, M = cost_matrix.size()

        K = torch.exp(-cost_matrix / self.epsilon)  # Gibbs kernel
        print("K1  : ", K)
        K = K / (torch.sum(K, dim=-1, keepdim=True) + 1e-16)  # Avoid division by zero
        print("K2  : ", K)
        u = torch.ones((B, N, 1), device=cost_matrix.device)  # [B, N, 1]
        v = torch.ones((B, M, 1), device=cost_matrix.device)  # [B, M, 1]

        for _ in range(self.max_iter):
            u = 1.0 / (K @ v)  # Update u
            v = 1.0 / (K.transpose(-2, -1) @ u)  # Update v

            # Clipping to avoid NaNs
            u = torch.clamp(u, min=1e-8, max=1e8)
            v = torch.clamp(v, min=1e-8, max=1e8)

        transport_matrix = u * K * v.transpose(-2, -1)  # [B, N, M]
        return transport_matrix
    
    def wasserstein_distance(self, p, q, transport_matrix):

        return torch.sum(transport_matrix * torch.cdist(p, q, p=2))

    def compute_wasserstein_distance(self, transport_matrix, cost_matrix):
        # Compute Wasserstein distance as the sum of element-wise products of transport and cost matrices
        wasserstein_dist = (transport_matrix * cost_matrix).sum(dim=(1, 2))  # [B] -> Scalar
        
        return wasserstein_dist.mean()

    def compute_wasserstein_distance(self, transport_matrix, cost_matrix):
        # Compute Wasserstein distance as the sum of element-wise products of transport and cost matrices
        wasserstein_dist = (transport_matrix * cost_matrix).sum(dim=(1, 2))  # [B, N, M] -> [B]
        return wasserstein_dist.mean()  # Mean over batch

    def compute_ot_loss(self, teacher_out, student_out):
        B, T_teacher, V_teacher = teacher_out.shape
        B, T_student, V_student = student_out.shape

        # Positional cost matrix
        pos_teacher = torch.arange(T_teacher, device=teacher_out.device).view(1, -1, 1).expand(B, -1, T_student)
        pos_student = torch.arange(T_student, device=student_out.device).view(1, 1, -1).expand(B, T_teacher, -1)
        # Compute the Euclidean cost matrix
        cost_matrix_euclidean = torch.cdist(
            teacher_out.view(B, -1, V_teacher), 
            student_out.view(B, -1, V_student), p=2
        )  # [B, T_teacher, T_student]
        print("cos euc : ", cost_matrix_euclidean.size(), cost_matrix_euclidean)
        cost_matrix_positional = self.gamma * torch.abs(pos_teacher - pos_student)  # [B, T_teacher, T_student]
        print("cos pos :", cost_matrix_positional.size(), cost_matrix_positional)
        # Combine Euclidean and positional cost
        total_cost_matrix = cost_matrix_euclidean + cost_matrix_positional
        print("tot cost :", total_cost_matrix.size(), total_cost_matrix)
        # Apply Sinkhorn-Knopp algorithm
        transport_matrix = self.sinkhorn_knopp(total_cost_matrix)
        print("trans : ", transport_matrix.size(), transport_matrix)

        # Compute Wasserstein distance
        wasserstein_dist = self.compute_wasserstein_distance(transport_matrix, total_cost_matrix)
        
        return wasserstein_dist
        
    def _calc_ot_loss(self, student_audio_out, teacher_audio_out, student_text_out, teacher_text_out):
        print("s a : ", student_audio_out.size())
        print("t a : ", teacher_audio_out.size())
        print("s t : ", student_text_out.size())
        print("t t : ", teacher_text_out.size())
        audio_loss = self.compute_ot_loss(teacher_audio_out, student_audio_out)
        print("audio : ", audio_loss)
        text_loss = self.compute_ot_loss(teacher_text_out, student_text_out)
        print("text : ", text_loss)

        return audio_loss + text_loss



    # def distance_with_position(self, p, q, positions_p, positions_q):
    #     euclidean_dist = torch.cdist(p, q, p=2)     # Compute pairwise Euclidean distance
    #     positions_p = positions_p.float()
    #     positions_q = positions_q.float()
    #     position_dist = torch.cdist(positions_p.unsqueeze(-1), positions_q.unsqueeze(-1), p=1)       # Compute position distance
    #     return euclidean_dist + self.lambda_pos * position_dist

    # def sinkhorn_knopp(self, cost_matrix):
    #     B, N, _ = cost_matrix.size()

    #     K = torch.exp(-cost_matrix / self.epsilon)  # Gibbs kernel
    #     K = K / (torch.sum(K, dim=-1, keepdim=True) + 1e-16)  # Avoid division by zero
    #     u = torch.ones((B, N, 1), device=cost_matrix.device)  # [B, N, 1]
    #     v = torch.ones((B, N, 1), device=cost_matrix.device)  # [B, N, 1]

    #     for _ in range(self.max_iter):
    #         u = 1.0 / (K @ v)  # Update u
    #         v = 1.0 / (K.transpose(-2, -1) @ u)  # Update v

    #         # Clipping to avoid NaNs
    #         u = torch.clamp(u, min=1e-8, max=1e8)
    #         v = torch.clamp(v, min=1e-8, max=1e8)

    #     transport_matrix = u * K * v.transpose(-2, -1)  # [B, N, N]
    #     return transport_matrix
    # def compute_wasserstein_distance(self, transport_matrix, cost_matrix):
    #     # Compute Wasserstein distance as the sum of element-wise products of transport and cost matrices
    #     wasserstein_dist = (transport_matrix * cost_matrix).sum(dim=(1, 2))  # [B, N, N] -> [B]
    #     return wasserstein_dist.mean()  # Mean over batch

    # def kl_divergence(self, p, q):
    #     return (p * (p / q).log()).sum(dim=-1)

    # def compute_positional_cost(self, B, T_chunk, U, device):
    #     # Compute positional cost based on Euclidean distance between positions and absolute difference in time
    #     pos_x = torch.arange(T_chunk, device=device).view(-1, 1).repeat(1, U)  # [chunk, U]
    #     pos_y = torch.arange(U, device=device).view(1, -1).repeat(T_chunk, 1)  # [chunk, U]

    #     pos_x_flat = pos_x.view(-1)  # [chunk*U]
    #     pos_y_flat = pos_y.view(-1)  # [chunk*U]

    #     pos = torch.stack((pos_x_flat, pos_y_flat), dim=1)  # [chunk*U, 2]

    #     # Compute pairwise Euclidean distance for u_i and v_j
    #     euclidean_cost = torch.cdist(pos.float(), pos.float(), p=2)  # [chunk*U, chunk*U]

    #     # Compute absolute difference in time for s_i and t_j
    #     time_diff = torch.abs(pos_x_flat[:, None] - pos_x_flat[None, :])  # [chunk*U, chunk*U]

    #     # Combine costs
    #     position_cost = euclidean_cost + time_diff  # [chunk*U, chunk*U]
        
    #     # Repeat for batch dimension
    #     position_cost = position_cost.view(1, T_chunk * U, T_chunk * U).repeat(B, 1, 1)  # [B, chunk*U, chunk*U]

    #     return position_cost

    # def optimal_transport_alignment(
    #     self,
    #     joint_out: torch.Tensor,
    #     teacher_joint_out: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Calculate the differences between teacher and student model's logits

    #     Args:
    #         joint_out: student model's joint output(logits). (B, T, U, V)
    #         teacher_joint_out: teacher model's joint output(logits). (B, T, U, V)

    #     """
        
    #     B, T, U, V = teacher_joint_out.shape

    #     total_loss = 0

    #     for t_start in range(0, T, self.chunk_size):
    #         t_end = min(t_start + self.chunk_size, T)

    #         teacher_probs = F.softmax(teacher_joint_out[:, t_start:t_end, :, :], dim=-1)  # [B, chunk, U, V]
    #         student_probs = F.softmax(joint_out[:, t_start:t_end, :, :], dim=-1)  # [B, chunk, U, V]

    #         # Compute the Euclidean cost matrix for the chunk
    #         cost_matrix_euclidean = torch.cdist(teacher_probs.view(B, (t_end - t_start) * U, V), 
    #                                             student_probs.view(B, (t_end - t_start) * U, V), p=2)  # [B, chunk*U, chunk*U]
    #         print("cost euc : ", cost_matrix_euclidean.size(), cost_matrix_euclidean)
    #         # Compute the positional cost matrix for the chunk
    #         position_cost = self.compute_positional_cost(B, t_end - t_start, U, teacher_joint_out.device)  # [B, chunk*U, chunk*U]
    #         print("pos : ", position_cost.size(), position_cost)
    #         # Combine the Euclidean cost matrix and positional cost matrix
    #         cost_matrix = cost_matrix_euclidean + position_cost  # [B, chunk*U, chunk*U]
    #         print("cost : ", cost_matrix.size(), cost_matrix)
    #         # Apply Sinkhorn-Knopp algorithm
    #         transport_matrix = self.sinkhorn_knopp(cost_matrix)
    #         print("trans : ", transport_matrix.size(), transport_matrix)
    #         # Compute Wasserstein distance for the chunk
    #         wasserstein_dist = self.compute_wasserstein_distance(transport_matrix, cost_matrix)
    #         print("was : ", wasserstein_dist.size(), wasserstein_dist)
    #         total_loss += wasserstein_dist
    #     #     # Compute the KL divergence cost matrix
    #     #     cost_matrix_kl = self.kl_divergence(student_probs, teacher_probs).squeeze()  # [B, T, U]
    #     #     print("kl : ", cost_matrix_kl.size())
    #     #     position_cost = self.compute_positional_cost(T, U, teacher_joint_out.device)  # [T, U, T, U]
    #     #     print("pos : ", position_cost.size())
    #     #     cost_matrix = cost_matrix_kl.unsqueeze(-1).unsqueeze(-1) + position_cost  # [B, T, U, T, U]
    #     #     print(" cost : ", cost_matrix.size())

    #     #     # Apply Sinkhorn-Knopp algorithm
    #     #     transport_matrix = self.sinkhorn_knopp(cost_matrix)

    #     #     # Compute Wasserstein distance
    #     #     wasserstein_dist = self.compute_wasserstein_distance(transport_matrix, cost_matrix)

    #     #     ot_loss += wasserstein_dist
    
    #     return total_loss


    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features sequences and features sequences lengths.

        Args:
            speech: Speech sequences. (B, S)
            speech_lengths: Speech sequences lengths. (B,)

        Return:
            feats: Features sequences. (B, T, D_feats)
            feats_lengths: Features sequences lengths. (B,)

        """
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            feats, feats_lengths = speech, speech_lengths

        return feats, feats_lengths

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        joint_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            joint_out: Joint Network output sequences (B, T, U, D_joint)
            target: Target label ID sequences. (B, L)
            t_len: Encoder output sequences lengths. (B,)
            u_len: Target label ID sequences lengths. (B,)

        Return:
            loss_transducer: Transducer loss value.

        """
        if self.criterion_transducer is None:
            try:
                from warprnnt_pytorch import RNNTLoss

                self.criterion_transducer = RNNTLoss(
                    reduction="mean",
                    fastemit_lambda=self.fastemit_lambda,
                )
            except ImportError:
                logging.error(
                    "warp-transducer was not installed. "
                    "Please consult the installation documentation."
                )
                exit(1)

        with autocast(False):
            loss_transducer = self.criterion_transducer(
                joint_out.float(),
                target,
                t_len,
                u_len,
            )

        return loss_transducer

    def _calc_kd_loss(
        self,
        joint_out,
        teacher_joint_out,
    ) -> torch.Tensor:
        """Compute Knowledge Distillation loss.

        Args:
            joint_out: Joint Network output sequences. (B, T, U, D_joint)
            teacher_joint_out: Teacher model's Joint Network output sequences. (B, T, U, D_joint)

        Return:
            loss_kd: Knowledge distillation loss value.
        """
        
        s_in = F.log_softmax(joint_out / self.temp_tau, dim=-1)
        t_in = F.log_softmax(teacher_joint_out / self.temp_tau, dim=-1)
        
        loss_kd = self.kl_loss(s_in, t_in)

        return loss_kd / 100

    def _calc_k2_transducer_pruned_loss(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        labels: torch.Tensor,
        encoder_out_len: torch.Tensor,
        decoder_out_len: torch.Tensor,
        prune_range: int = 5,
        simple_loss_scaling: float = 0.5,
        lm_scale: float = 0.0,
        am_scale: float = 0.0,
        loss_type: str = "regular",
        reduction: str = "mean",
        padding_idx: int = 0,
    ) -> torch.Tensor:
        """Compute k2 pruned Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            decoder_out: Decoder output sequences. (B, T, D_dec)
            labels: Label ID sequences. (B, L)
            encoder_out_len: Encoder output sequences lengths. (B,)
            decoder_out_len: Target label ID sequences lengths. (B,)
            prune_range: How many tokens by frame are used compute the pruned loss.
            simple_loss_scaling: The weight to scale the simple loss after warm-up.
            lm_scale: The scale factor to smooth the LM part.
            am_scale: The scale factor to smooth the AM part.
            loss_type: Define the type of path to take for loss computation.
                         (Either 'regular', 'smoothed' or 'constrained')
            padding_idx: SOS/EOS + Padding index.

        Return:
            loss_transducer: Transducer loss value.

        """
        try:
            import k2

            if self.fastemit_lambda > 0.0:
                logging.info(
                    "Disabling FastEmit, it is not available with k2 Transducer loss. "
                    "Please see delay_penalty option instead."
                )
        except ImportError:
            logging.error(
                "k2 was not installed. Please consult the installation documentation."
            )
            exit(1)

        # Note (b-flo): We use a dummy scaling scheme until the training parts are
        # revised (in a short future).
        self.steps_num += 1

        if self.steps_num < self.warmup_steps:
            pruned_loss_scaling = 0.1 + 0.9 * (self.steps_num / self.warmup_steps)
            simple_loss_scaling = 1.0 - (
                (self.steps_num / self.warmup_steps) * (1.0 - simple_loss_scaling)
            )
        else:
            pruned_loss_scaling = 1.0

        labels_unpad = [y[y != self.ignore_id].tolist() for y in labels]

        target = k2.RaggedTensor(labels_unpad).to(decoder_out.device)
        target_padded = target.pad(mode="constant", padding_value=padding_idx)
        target_padded = target_padded.to(torch.int64)

        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = decoder_out_len
        boundary[:, 3] = encoder_out_len

        lm = self.lm_proj(decoder_out)
        am = self.am_proj(encoder_out)

        with autocast(False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm.float(),
                am.float(),
                target_padded,
                padding_idx,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                rnnt_type=loss_type,
                reduction=reduction,
                return_grad=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad,
            py_grad,
            boundary,
            prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            self.joint_network.lin_enc(encoder_out),
            self.joint_network.lin_dec(decoder_out),
            ranges,
        )

        joint_out = self.joint_network(am_pruned, lm_pruned, no_projection=True)

        with autocast(False):
            pruned_loss = k2.rnnt_loss_pruned(
                joint_out.float(),
                target_padded,
                ranges,
                padding_idx,
                boundary,
                rnnt_type=loss_type,
                reduction=reduction,
            )

        loss_transducer = (
            simple_loss_scaling * simple_loss + pruned_loss_scaling * pruned_loss
        )

        return loss_transducer

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)
            t_len: Encoder output sequences lengths. (B,)
            u_len: Target label ID sequences lengths. (B,)

        Return:
            loss_ctc: CTC loss value.

        """
        ctc_in = self.ctc_lin(
            torch.nn.functional.dropout(encoder_out, p=self.ctc_dropout_rate)
        )
        ctc_in = torch.log_softmax(ctc_in.transpose(0, 1), dim=-1)

        target_mask = target != 0
        ctc_target = target[target_mask].cpu()

        with torch.backends.cudnn.flags(deterministic=True):
            loss_ctc = torch.nn.functional.ctc_loss(
                ctc_in,
                ctc_target,
                t_len,
                u_len,
                zero_infinity=True,
                reduction="sum",
            )
        loss_ctc /= target.size(0)

        return loss_ctc

    def _calc_lm_loss(
        self,
        decoder_out: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LM loss (i.e.: Cross-entropy with smoothing).

        Args:
            decoder_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)

        Return:
            loss_lm: LM loss value.

        """
        batch_size = decoder_out.size(0)

        logp = torch.log_softmax(
            self.lm_lin(decoder_out[:, :-1, :]).view(-1, self.vocab_size),
            dim=1,
        )
        target = target.view(-1).type(torch.int64)
        ignore = (target == 0).unsqueeze(1)

        with torch.no_grad():
            true_dist = logp.clone().fill_(self.lm_loss_smooth_neg)

            true_dist.scatter_(1, target.unsqueeze(1), self.lm_loss_smooth_pos)

        loss_lm = torch.nn.functional.kl_div(logp, true_dist, reduction="none")
        loss_lm = loss_lm.masked_fill(ignore, 0).sum() / batch_size

        return loss_lm
