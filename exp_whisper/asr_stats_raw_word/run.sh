./asr_lm_multi.sh --ngpu 1 --stage 10 --stop_stage 11 --token_type word --expdir exp_whisper --max_wav_duration 30 --asr_config conf/gop/train_whisper_feats_gop.yaml --inference_config conf/tuning/transducer/decode.yaml --train_set speechocean_train --valid_set speechocean_test --test_sets speechocean_test --lm_train_text 'data/speechocean_train/text data/local/other_text/text' --bpe_train_text data/speechocean_train/text --stage 10 "$@"; exit $?