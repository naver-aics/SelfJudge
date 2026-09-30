# SelfJudge
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# Derived from garipovroma/autojudge (Apache-2.0):
#   https://github.com/garipovroma/autojudge/blob/cf180e89a8718e10a34ccf222a2a96e786e24a20/scripts/find_important_tokens_lcb.sh
# Modifications for SelfJudge:
#   - runs src/compare_important_tokens_lcb.py (SelfJudge likelihood-difference
#     labeling) with --num_next_token instead of src/find_important_tokens_lcb.py.

# export MODEL0="meta-llama/Llama-3.2-1B-Instruct"
# export MODEL1="meta-llama/Llama-3.1-8B-Instruct"

export MODEL0="Qwen/Qwen2.5-0.5B-Instruct"
export MODEL1="Qwen/Qwen2.5-7B-Instruct"


# output='compare_lcb'
output='compare_lcb_qwen'

export TORCH_DTYPE=auto
export RANDOM_SEED=42
export MAX_NEW_TOKENS=2048
export OUTPUT_FOLDER="$output"

export N_TASKS=220
export OUTPUT_FILE="important_tokens--model-selfjudge_nextfew_all-${N_TASKS}_samples"

export DUMP_FREQ=64

echo "OUTPUT_FILE IS $OUTPUT_FILE"

mkdir $OUTPUT_FOLDER

CUDA_VISIBLE_DEVICES=2 python3 src/compare_important_tokens_lcb.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --random_seed $RANDOM_SEED \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_folder $OUTPUT_FOLDER \
    --output_file $OUTPUT_FILE \
    --dump_freq $DUMP_FREQ \
    --n_tasks $N_TASKS