#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

<<<<<<< HEAD:egs2/librispeech/asr1/run_kd.sh
asr_task="asr_transducer_kd"
asr_config="conf/tuning/transducer/KD/train_conformer-statelesst-streaming-kd.yaml"
inference_config=conf/decode_asr_rnnt.yaml
=======
asr_task="asr_transducer_wkd"
asr_config="conf/tuning/transducer/KD/train_asr_conformer-rnnt-streaming-kd.yaml"
inference_config="conf/tuning/transducer/decode_asr_transducer.yaml"
asr_exp="exp/asr_train_conformer-statelesst-streaming_baseline_sp_wkd"
>>>>>>> 3361fbc56e97029d64dd5321e32e834a7f900bbd:egs2/librispeech_100/asr1/run_kd.sh

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_task "${asr_task}" \
<<<<<<< HEAD:egs2/librispeech/asr1/run_kd.sh
=======
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_exp "${asr_exp}" \
>>>>>>> 3361fbc56e97029d64dd5321e32e834a7f900bbd:egs2/librispeech_100/asr1/run_kd.sh
    --asr_config "${asr_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
