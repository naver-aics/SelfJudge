# SelfJudge
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# Derived from garipovroma/autojudge (Apache-2.0):
#   https://github.com/garipovroma/autojudge/blob/cf180e89a8718e10a34ccf222a2a96e786e24a20/src/find_important_tokens_lcb.py
# Modifications for SelfJudge:
#   - mines mismatches on open-ended Dolly-15k instructions (HF `treadon/dolly-15k`)
#     where no test-based correctness label exists: the pass/fail check is dropped
#     (label = None) and only the SelfJudge likelihood differences over the next
#     0/5/10/20 tokens are stored.
#   - chat-template prompt formatting; Qwen2.5 tokenizer handling and defaults.

import argparse
import os
import time
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import itertools
import logging

from utils import color_replaced_tokens
from datasets import load_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

prompt_with_0_shots = "Given the following problem, reason and give a final answer to the problem.\n"
formatting_prompt = "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."


def verify_args(args):
    assert args.process_id < args.world_size, "--process_id must be < --world_size"


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for important tokens selection algorithm")

    parser.add_argument('--draft_model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help='Path or identifier of the draft model.')
    parser.add_argument('--target_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Path or identifier of the target model.')
    parser.add_argument('--torch_dtype', type=str, choices=['float32', 'auto'], default='auto',
                        help='Data type for torch tensors.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_folder', type=str, default='src/selfjudge_dolly_qwen',
                        help='Output folder name.')
    parser.add_argument('--output_file', type=str, default='important_tokens',
                        help='Output file name.')
    parser.add_argument('--world_size', type=int, default=1,
                        help='world size')
    parser.add_argument('--process_id', type=int, default=0,
                        help='Process ID')
    parser.add_argument('--dump_freq', type=int, default=100,
                        help='Dump frequency.')
    parser.add_argument('--write_index_offset', type=int, default=0,
                        help='Write index offset.')
    parser.add_argument('--keep_responses', type=bool, default=False,
                        help='Keep all responses in find_important_tokens algorithm output.')
    parser.add_argument('--n_tasks', type=int, default=32)
    parser.add_argument('--num_process_evaluate', type=int, default=16)
    parser.add_argument('--process_saver_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=8)
    parser.add_argument('--lcb_path', type=str, default='none')

    args = parser.parse_args()
    verify_args(args)

    return args


@torch.no_grad()
def find_important_tokens(
        prompt: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizer,
        draft_model: transformers.AutoModelForCausalLM,
        target_model: transformers.AutoModelForCausalLM,
        sample: dict,
        ):
    batch_size, prompt_max_length = prompt.shape
    assert batch_size == 1

    device = prompt.device

    batch_dict = {
        'input_ids': prompt,
        'attention_mask': torch.ones_like(prompt),
    }

    target_response = target_model.generate(
        **batch_dict,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_length=prompt_max_length + args.max_new_tokens
    )

    outputs = target_model.forward(input_ids=target_response,
                                      attention_mask=torch.ones_like(target_response),
                                      output_hidden_states=True)

    logp_orig = F.log_softmax(outputs.logits, dim=-1)

    current_response = target_response  # to be updated by algorithm

    changed_token_indices = []
    colored_tokens = None
    left_border = prompt_max_length

    responses = [(current_response.cpu().clone())]

    draft_argmax_tokens = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device=device),
                                         draft_model.forward(input_ids=current_response, attention_mask=torch.ones_like(
                                             current_response)).logits.argmax(-1)[:, :-1]], dim=1)

    likelihood_diffs_0 = []
    likelihood_diffs_5 = []
    likelihood_diffs_10 = []
    likelihood_diffs_20 = []

    for iteration in itertools.count():

        is_mismatch = current_response != draft_argmax_tokens  # [batch_size=1, max_length]
        mismatch_indices = sorted([idx for idx in is_mismatch.flatten().nonzero().flatten().tolist() if (idx >= left_border) and (idx < len(current_response[0])-1)])

        if not mismatch_indices:
            break

        colored_tokens = color_replaced_tokens(current_response, draft_argmax_tokens, prompt_max_length,
                                               changed_token_indices, tokenizer)

        mismatch_index = mismatch_indices[0]

        prefix_with_draft_token = torch.cat([
              current_response[:, :mismatch_index],
              torch.tensor([[draft_argmax_tokens[:, mismatch_index]]], device=device),
              current_response[:, mismatch_index+1:]
            ], dim=1)

        alternative_outputs = target_model.forward(input_ids=prefix_with_draft_token,
                                      attention_mask=torch.ones_like(prefix_with_draft_token),
                                      output_hidden_states=True)


        if mismatch_index >= len(current_response[0]):
            break

        # Compute P(future|match(orig), prefix) num_next = 0
        num_next_token = 0
        if mismatch_index + num_next_token < logp_orig.shape[1]:
            joint_logp_orig_0 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:mismatch_index+num_next_token], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_0 = torch.max(logp_alter[0, mismatch_index:mismatch_index+num_next_token], dim=-1)[0]
            joint_logp_alter_0 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_0])).item()
        else:
            # Compute P(future|mismatch(alter), prefix)
            joint_logp_orig_0 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:-1], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_0 = torch.max(logp_alter[0, mismatch_index:-1], dim=-1)[0]
            joint_logp_alter_0 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_0])).item()


        # Compute P(future|match(orig), prefix) num_next = 5
        num_next_token = 5
        if mismatch_index + num_next_token < logp_orig.shape[1]:
            joint_logp_orig_5 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:mismatch_index+num_next_token], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_5 = torch.max(logp_alter[0, mismatch_index:mismatch_index+num_next_token], dim=-1)[0]
            joint_logp_alter_5 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_5])).item()
        else:
            # Compute P(future|mismatch(alter), prefix)
            joint_logp_orig_5 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:-1], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_5 = torch.max(logp_alter[0, mismatch_index:-1], dim=-1)[0]
            joint_logp_alter_5 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_5])).item()

        # Compute P(future|match(orig), prefix) num_next = 10
        num_next_token = 10
        if mismatch_index + num_next_token < logp_orig.shape[1]:
            joint_logp_orig_10 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:mismatch_index+num_next_token], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_10 = torch.max(logp_alter[0, mismatch_index:mismatch_index+num_next_token], dim=-1)[0]
            joint_logp_alter_10 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_10])).item()
        else:
            # Compute P(future|mismatch(alter), prefix)
            joint_logp_orig_10 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:-1], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_10 = torch.max(logp_alter[0, mismatch_index:-1], dim=-1)[0]
            joint_logp_alter_10 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_10])).item()

        # Compute P(future|match(orig), prefix) num_next = 0
        num_next_token = 20
        if mismatch_index + num_next_token < logp_orig.shape[1]:
            joint_logp_orig_20 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:mismatch_index+num_next_token], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_20 = torch.max(logp_alter[0, mismatch_index:mismatch_index+num_next_token], dim=-1)[0]
            joint_logp_alter_20 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_20])).item()
        else:
            # Compute P(future|mismatch(alter), prefix)
            joint_logp_orig_20 = torch.mean(torch.max(logp_orig[0, mismatch_index-1:-1], dim=-1)[0]).item() ##
            logp_alter = F.log_softmax(alternative_outputs.logits, dim=-1)
            joint_logp_alter_20 = torch.max(logp_alter[0, mismatch_index:-1], dim=-1)[0]
            joint_logp_alter_20 = torch.mean(torch.cat([logp_alter[0, mismatch_index-1, draft_argmax_tokens[:, mismatch_index]], joint_logp_alter_20])).item()


        changed_token_indices.append((mismatch_index, None, current_response[0, mismatch_index].item(),
                                        torch.tensor([[draft_argmax_tokens[:, mismatch_index]]]).item()))

        print(f'Processing Remaining: {len(mismatch_indices)}')

        left_border = mismatch_index + 1
        responses.append((current_response.cpu().clone(), None))
        likelihood_diffs_0.append(joint_logp_orig_0 - joint_logp_alter_0)
        likelihood_diffs_5.append(joint_logp_orig_5 - joint_logp_alter_5)
        likelihood_diffs_10.append(joint_logp_orig_10 - joint_logp_alter_10)
        likelihood_diffs_20.append(joint_logp_orig_20 - joint_logp_alter_20)

    return dict(changed_token_indices=changed_token_indices, colored_tokens=colored_tokens,
                current_response=current_response, responses=responses, likelihood_diffs_0=likelihood_diffs_0,
                likelihood_diffs_5=likelihood_diffs_5, likelihood_diffs_10=likelihood_diffs_10, likelihood_diffs_20=likelihood_diffs_20)


if __name__ == '__main__':
    args = get_args()
    print(f'The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))

    np.random.seed(args.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    draft_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=args.torch_dtype, device_map=device, low_cpu_mem_usage=True)
    if '70b' in args.target_model.lower():
        device_map = 'auto'
    else:
        device_map = device
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=args.torch_dtype, device_map=device_map, low_cpu_mem_usage=True)
    tokenizer_path = args.target_model.split('/snapshots')[0].replace('models', 'tokenizers')
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left')

    if ('Llama-3' in args.target_model) or ('llama-3' in args.target_model):
        tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>

    if ('Qwen2.5' in args.target_model) or ('qwen2.5' in args.target_model):
        tokenizer.bos_token_id = tokenizer.encode('<|im_start|>')[0]

    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("treadon/dolly-15k", split="train", cache_dir='data/dolly15k')

    n_samples = len(dataset)

    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)
    logger.info(f'{n_samples=}, {args.world_size=}, {shard_len=}, {shard_start=}, {shard_end=}')
    logger.info(f'Process {args.process_id} will process {shard_end - shard_start} questions: [{shard_start}; {shard_end})')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f'Created output folder {args.output_folder}')
    else:
        logger.info(f'Output folder {args.output_folder} already exists')

    output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_id}.pt')
    if os.path.exists(output_file_path):
        mined_data = torch.load(output_file_path)
        logger.info(f'Loaded {len(mined_data)} important tokens from {output_file_path}, starting from {shard_start + len(mined_data)}-th sample')
    else:
        mined_data = []
        logger.info(f'No important tokens found in {output_file_path}, will generate them')

    mined_data = []
    sample_idx = 0
    with tqdm(total=shard_end - shard_start) as pbar:
        pbar.update(len(mined_data))
        for sample_idx in range(shard_start + len(mined_data), shard_end):

            question = dataset[sample_idx]['instruction']
            prompt = question

            batch_input_ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=True, return_tensors='pt', padding=True, continue_final_message=False # <--- MIGHT BE A BUG
            ).to(device)

            important_tokens_dict = find_important_tokens(batch_input_ids, tokenizer, draft_model, target_model, dataset[sample_idx])
            important_tokens_dict['id'] = sample_idx
            mined_data.append(important_tokens_dict)


            if ((sample_idx % args.dump_freq == (args.dump_freq - 1)) or (sample_idx == shard_end - 1)):
                torch.save(mined_data, output_file_path)
            pbar.update(1)
            pbar.set_description(f'Process {args.process_id}')

    done_file = os.path.join(args.output_folder, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")
    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == args.process_saver_id and args.world_size > 1:
        logger.info(f"Process {args.process_saver_id} is waiting for all other processes to finish...")
        while True:
            done_files = [f"done_{i}.txt" for i in range(args.process_saver_id + 1, args.process_saver_id + args.total_gpus) if i * shard_len < shard_end]
            all_done = all(os.path.exists(os.path.join(args.output_folder, f)) for f in done_files)

            if all_done:
                logger.info("All processes finished. Concatenating important tokens data...")
                all_data = []
                for i in range(args.process_saver_id, args.process_saver_id + args.total_gpus):
                    if i * shard_len >= shard_end:
                        continue
                    process_output_file = os.path.join(args.output_folder, f"{args.output_file}_{i}.pt")
                    all_data.extend(torch.load(process_output_file))
                    logger.info(f"Removing {process_output_file}")
                    os.remove(process_output_file)

                    done_file = os.path.join(args.output_folder, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                final_output_file_path = os.path.join(args.output_folder, args.output_file + f'_{args.process_saver_id}' + '.pt')
                logger.info(f"Saving important tokens data to {final_output_file_path}...")
                torch.save(all_data, final_output_file_path)
                logger.info("Done.")
                break

            time.sleep(5)