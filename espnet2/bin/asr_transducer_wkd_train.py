#!/usr/bin/env python3

from espnet2.tasks.asr_transducer_wkd import ASRTransducerTask


def get_parser():
    """Get parser for ASR Transducer task."""
    parser = ASRTransducerTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR Transducer KD training.

    Example:

        % python asr_transducer_kd_train.py asr --print_config \
                --optim adadelta > conf/train_asr.yaml
        % python asr_transducer_kd_train.py \
                --config conf/tuning/transducer/KD/train_rnn_transducer.yaml
    """
    ASRTransducerTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
