#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other"
# test_sets="samples"

asr_task="asr_transducer_otg"
asr_config="conf/tuning/transducer/OT/conformer-rnnt-otg.yaml"
inference_config=conf/decode_asr_rnnt.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 2048 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_task "${asr_task}" \
    --asr_config "${asr_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \