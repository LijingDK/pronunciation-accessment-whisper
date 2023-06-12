#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=11
stop_stage=11

train_set="speechocean_train"
valid_set="speechocean_test"
test_sets="speechocean_test"

token_type=word
expdir=exp_whisper

#asr_config=conf/tuning/transducer/train_conformer-rnn_transducer.yaml
#asr_config=conf/gop/train_conformer_gop.yaml
#asr_config=conf/gop/train_transformer_gop.yaml
#asr_config=conf/gop/train_wavlm_feats_trans_gop.yaml

 asr_config=conf/gop/train_whisper_encoder_gop.yaml
#asr_config=conf/gop/whisper_wo_cnn_25.yaml
inference_config=conf/tuning/transducer/decode.yaml

./asr_lm_multi_whisper.sh \
    --ngpu 1 \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --token_type "${token_type}" \
    --expdir "${expdir}" \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
