#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/AED/asr_whisper.yaml
asr_task="asr_transducer"
lm_config=conf/tuning/train_lm_transformer2.yaml
# inference_config=conf/tuning/transducer/decode_asr_transducer.yaml
inference_config=conf/decode_asr.yaml
asr_exp=exp/whisper_v.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --token_type whisper_en \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --asr_exp "${asr_exp}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
