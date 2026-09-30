# Derived from garipovroma/autojudge (Apache-2.0):
#   https://github.com/garipovroma/autojudge/blob/cf180e89a8718e10a34ccf222a2a96e786e24a20/scripts/calc_hiddens.sh
# Modifications for SelfJudge:
#   - defaults point to the Qwen2.5 pair and SelfJudge output paths; the
#     multi-process example is commented out.

# export MODEL0="meta-llama/Llama-3.2-1B-Instruct"
# export MODEL1="meta-llama/Llama-3.1-8B-Instruct"

export MODEL0="Qwen/Qwen2.5-0.5B-Instruct"
export MODEL1="Qwen/Qwen2.5-7B-Instruct"


export TORCH_DTYPE=auto
export BATCH_SIZE=8

# export DATA_FILE="compare_lcb/important_tokens--model-selfjudge_nextfew_all-220_samples_0.pt"
# export OUTPUT_PATH="compare_lcb/important_tokens--model-selfjudge_nextfew_all-220_samples_0_with_hidden_new"

# gsm qwen
export DATA_FILE="compare_qwen/important_tokens--model-selfjudge-500_samples.pt"
export OUTPUT_PATH="compare_qwen/important_tokens--model-selfjudge-500_samples-hiddens"

# lcb qwen
# export DATA_FILE="compare_lcb_qwen/important_tokens--model-selfjudge_nextfew_all-220_samples_0.pt"
# export OUTPUT_PATH="compare_lcb_qwen/important_tokens--model-selfjudge_nextfew_all-220_samples_0_with_hidden-new"

n_samples=-1

export SAVE_FREQ=20
export N_PROCESSES=20
export PROCESS_ID=0
GPU=7

# single gpu run
CUDA_VISIBLE_DEVICES=$GPU python src/calc_hiddens.py \
    --draft_model $MODEL0 \
    --target_model $MODEL1 \
    --torch_dtype $TORCH_DTYPE \
    --batch_size $BATCH_SIZE \
    --data_file $DATA_FILE \
    --output_path $OUTPUT_PATH \
    --save_freq $SAVE_FREQ \
    --n_processes $N_PROCESSES \
    --process_id $PROCESS_ID \
    --n_samples $n_samples


# multiple gpus run
# export N_PROCESSES=2

# CUDA_VISIBLE_DEVICES=0 python src/calc_hiddens.py \
#     --draft_model $MODEL0 \
#     --target_model $MODEL1 \
#     --torch_dtype $TORCH_DTYPE \
#     --batch_size $BATCH_SIZE \
#     --data_file $DATA_FILE \
#     --output_path $OUTPUT_PATH \
#     --save_freq $SAVE_FREQ \
#     --n_processes $N_PROCESSES \
#     --process_id 0 &
# CUDA_VISIBLE_DEVICES=1 python src/calc_hiddens.py \
#     --draft_model $MODEL0 \
#     --target_model $MODEL1 \
#     --torch_dtype $TORCH_DTYPE \
#     --batch_size $BATCH_SIZE \
#     --data_file $DATA_FILE \
#     --output_path $OUTPUT_PATH \
#     --save_freq $SAVE_FREQ \
#     --n_processes $N_PROCESSES \
#     --process_id 1 &
# wait