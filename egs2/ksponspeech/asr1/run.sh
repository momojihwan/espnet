#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="eval_clean eval_other"
asr_task="asr_transducer"
asr_config="conf/tuning/transducer/train_asr_conformer-rnnt.yaml"
lm_config="conf/train_lm_transformer.yaml"
inference_config="conf/tuning/transducer/decode_asr_transducer.yaml"
nbpe=2309

./asr.sh \
    --lang kr \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --max_wav_duration 30 \
    --asr_task "${asr_task}" \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --nbpe "${nbpe}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" \
    --bpe_nlsyms '[unk]' "$@"