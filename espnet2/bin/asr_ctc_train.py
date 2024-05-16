#!/usr/bin/env python3
import os
from espnet2.tasks.asr_ctc import ASRCTCTask


def get_parser():
    """Get parser for ASR Transducer task."""
    parser = ASRCTCTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR Transducer training.

    Example:

        % python asr_transducer_train.py asr --print_config \
                --optim adadelta > conf/train_asr.yaml
        % python asr_transducer_train.py \
                --config conf/tuning/transducer/train_rnn_transducer.yaml
    """
    ASRCTCTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
