# SelfJudge
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# Derived from garipovroma/autojudge (Apache-2.0):
#   https://github.com/garipovroma/autojudge/blob/cf180e89a8718e10a34ccf222a2a96e786e24a20/src/calc_hiddens.py
# Modifications for SelfJudge:
#   - extracts target-model hidden states only (no draft-model hiddens): the hidden at
#     each mismatch position and at the position before it, for both the
#     draft-substituted and the original target response, plus the hidden at the end
#     of the user prompt (target_hiddens*, target_prev_hiddens*, prompt_hidden).
#   - added find_user_prompt_end_index_from_tokens() for the Llama-3 and Qwen2.5 chat
#     formats, and Qwen2.5 tokenizer BOS handling.
#   - defaults switched to the Qwen2.5 draft/target pair and SelfJudge data paths.

import argparse
import os

from tqdm.auto import tqdm
import numpy as np

import torch

import transformers
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def find_user_prompt_end_index_from_tokens(response_tokens, tokenizer, model_class='llama3'):
    """
    Return the index at which the user prompt ends in a sequence of token IDs.

    Args:
        response_tokens (list): sequence of token IDs
        tokenizer: tokenizer object (able to encode/decode)

    Returns:
        int: token index where the user prompt ends (-1 if not found)
    """

    if model_class == 'llama3':
        try:
            start_header_id = tokenizer.encode('<|start_header_id|>', add_special_tokens=False)[0]
            end_header_id = tokenizer.encode('<|end_header_id|>', add_special_tokens=False)[0]
            eot_id = tokenizer.encode('<|eot_id|>', add_special_tokens=False)[0]
            user_id = tokenizer.encode('assistant', add_special_tokens=False)[0]
        except:
            # Fall back to an alternative approach
            special_tokens = tokenizer.get_vocab()
            start_header_id = special_tokens.get('<|start_header_id|>')
            end_header_id = special_tokens.get('<|end_header_id|>')
            eot_id = special_tokens.get('<|eot_id|>')
            user_id = special_tokens.get('user')

            if any(token_id is None for token_id in [start_header_id, end_header_id, eot_id, user_id]):
                raise ValueError("Required special tokens not found")

        last_user_start = -1

        for i in range(len(response_tokens) - 2):
            if (response_tokens[i] == start_header_id and
                response_tokens[i + 1] == user_id and
                response_tokens[i + 2] == end_header_id):
                last_user_start = i

        if last_user_start == -1:
            AssertionError


        return last_user_start

    elif model_class == 'qwen2.5':
        try:
            start_header_id = tokenizer.encode('<|im_start|>', add_special_tokens=False)[0]
            end_header_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)[0]
            eot_id = tokenizer.encode('<|endoftext|>', add_special_tokens=False)[0]
            user_id = tokenizer.encode('assistant', add_special_tokens=False)[0]
        except:
            # Fall back to an alternative approach
            special_tokens = tokenizer.get_vocab()
            im_start = special_tokens.get('<|im_start|>')
            im_end = special_tokens.get('<|im_end|>')
            endoftext = special_tokens.get('<|endoftext|>')
            assistant_id = special_tokens.get('assistant')

            if any(token_id is None for token_id in [im_start, im_end, endoftext, assistant_id]):
                raise ValueError("Required special tokens not found")

        last_user_start = -1

        for i in range(len(response_tokens) - 1):
            if (response_tokens[i] == start_header_id and
                response_tokens[i + 1] == user_id):
                last_user_start = i

        if last_user_start == -1:
            AssertionError

        return last_user_start

    # Get the IDs of the special tokens
    # Find the last user section
    # Pattern: [start_header_id, user_id, end_header_id]


def verify_args(args):
    assert args.process_id < args.n_processes, "--process_id must be < --n_processes"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft_model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--target_model', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--torch_dtype', type=str, default='float16')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_file', type=str, default="./compare_qwen/important_tokens--model-selfjudge-500_samples.pt")
    parser.add_argument('--output_path', type=str, default="./compare_qwen/important_tokens--model-selfjudge-500_samples-with-hiddens")
    parser.add_argument('--process_id', type=int, default=0)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=128)
    parser.add_argument('--n_samples', type=int, default=-1)

    args = parser.parse_args()


    verify_args(args)

    return args

def save_checkpoint(checkpoint, checkpoint_path):
    torch.save(checkpoint, checkpoint_path)

if __name__ == '__main__':
    args = get_args()
    print(f'The script was run in the following way:')
    print("python script.py \\\n" + "\n".join(f"\t\t--{k} {v} \\" for k, v in vars(args).items()))


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
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, padding_side='right')

    if ('Llama-3' in args.target_model) or ('llama-3' in args.target_model):
        tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>

    if ('Qwen2.5' in args.target_model) or ('qwen2.5' in args.target_model):
        tokenizer.bos_token_id = tokenizer.encode('<|im_start|>')[0]


    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    try:
        data = torch.load(args.data_file)
    except FileNotFoundError as e:
        file_found = False
        for i in range(100):
            fixed_path = args.data_file.split('.pt')[0] + f'_{i}.pt'
            try:
                data = torch.load(fixed_path)
                file_found = True
                break
            except FileNotFoundError:
                continue
        if not file_found:
            raise e


    if args.n_samples != -1:
        data = data[:args.n_samples]
        logger.info(f'But args.n_samples={args.n_samples}')
    n = len(data)

    block_size = (n + args.n_processes - 1) // args.n_processes
    start = args.process_id * block_size
    end = min((args.process_id + 1) * block_size, n)

    data = data[start:end]
    logger.info(f'Process {args.process_id} has {end - start} samples, [{start}:{end})')

    for idx, sample_dict in enumerate(data):
        sample_dict['id'] = idx
        sample_dict['target_hiddens'] = []
        sample_dict['target_prev_hiddens'] = []
        sample_dict['target_hiddens_replaced'] = []
        sample_dict['target_prev_hiddens_replaced'] = []
        sample_dict['prompt_hidden'] = []

    tokens_to_encode = []
    text_sample_ids = []
    mismatch_ids = []
    prompt_ending_ids = []

    n_samples = len(data)

    for sample_idx in tqdm(range(n_samples), total=n_samples, desc=f'Process {args.process_id}/{args.n_processes}'):
        sample_dict = data[sample_idx]

        token_ids = sample_dict['current_response']
        model_class = 'llama3' if ('llama3' in args.target_model) or ('Llama3' in args.target_model) else 'qwen2.5'
        prompt_end_idx = find_user_prompt_end_index_from_tokens(token_ids[0], tokenizer, model_class)


        for mismatch_idx, (changed_token_pos, importance, target_token, draft_token) in enumerate(data[sample_idx]['changed_token_indices']):
            orig_token = token_ids[:, changed_token_pos].item()
            replacement_token = draft_token

            token_ids[:, changed_token_pos] = draft_token
            tokens_to_encode.append(token_ids.clone()) # even batch indices hold the replaced response
            text_sample_ids.append(sample_idx)
            mismatch_ids.append(mismatch_idx)
            prompt_ending_ids.append(prompt_end_idx)

            token_ids[:, changed_token_pos] = target_token
            tokens_to_encode.append(token_ids.clone()) # odd batch indices hold the target response
            text_sample_ids.append(sample_idx)
            mismatch_ids.append(mismatch_idx)
            prompt_ending_ids.append(prompt_end_idx)

    n_seqs_to_encode = len(tokens_to_encode)
    n_tokens = 0

    orig_output_path = args.output_path
    args.output_path = f'{args.output_path}_{args.process_id}.pt'

    # check if args.output_path exists
    loaded_from_checkpoint = False
    loaded_checkpoint_batch_end = None
    if os.path.exists(args.output_path):
        checkpoint = torch.load(args.output_path)
        data = checkpoint['data']
        logger.info(f'File {args.output_path} exist')
        logger.info(f'Loaded {len(data)} samples from {args.output_path}')
        loaded_from_checkpoint = True
        loaded_checkpoint_batch_end = checkpoint['last_batch_end']
        logger.info(f'Loaded checkpoint batch end: {loaded_checkpoint_batch_end}')
    else:
        logger.info(f'File {args.output_path} does not exist, starting from scratch')

    for i in data:
        if 'responses' in i:
            del i['responses']


    iter_id = 0
    for batch_start in tqdm(range(0, n_seqs_to_encode, args.batch_size), desc=f'Process {args.process_id}/{args.n_processes}'):

        iter_id += 1
        batch_end = min(n_seqs_to_encode, batch_start + args.batch_size)
        max_seq_len = max(i.numel() for i in tokens_to_encode[batch_start:batch_end])

        if loaded_from_checkpoint:
            if batch_start < loaded_checkpoint_batch_end:
                continue


        padded_token_ids_list = []
        # input_ids padding

        tokens_to_pad = [tokens_to_encode[sample_idx].flatten() for sample_idx in range(batch_start, batch_end)]

        inputs = tokenizer.pad(dict(input_ids=tokens_to_pad), return_tensors='pt')
        inputs = inputs.to(device)

        # draft & target forward to get hiddens
        with torch.no_grad():
            target_hiddens = target_model.model(**inputs).last_hidden_state.cpu()

        del inputs

        for sample_in_batch_idx in range(0, batch_end - batch_start):
            sample_idx = batch_start + sample_in_batch_idx
            mismatch_idx = mismatch_ids[sample_idx]
            text_sample_idx = text_sample_ids[sample_idx]
            prompt_sample_idx = prompt_ending_ids[sample_idx]

            changed_token_pos, _, _, _ = data[text_sample_idx]['changed_token_indices'][mismatch_idx]

            target_hidden = target_hiddens[sample_in_batch_idx, changed_token_pos].to(torch.float32)

            target_prev_hidden = target_hiddens[sample_in_batch_idx, changed_token_pos - 1].to(torch.float32)

            prompt_hidden = target_hiddens[sample_in_batch_idx, prompt_sample_idx].to(torch.float32)

            if sample_in_batch_idx % 2 == 0:
                data[text_sample_ids[sample_idx]]['target_hiddens_replaced'].append(target_hidden)
                data[text_sample_ids[sample_idx]]['target_prev_hiddens_replaced'].append(target_prev_hidden)
                data[text_sample_ids[sample_idx]]['prompt_hidden'].append(prompt_hidden)
            else:
                data[text_sample_ids[sample_idx]]['target_hiddens'].append(target_hidden)
                data[text_sample_ids[sample_idx]]['target_prev_hiddens'].append(target_prev_hidden)
                data[text_sample_ids[sample_idx]]['prompt_hidden'].append(prompt_hidden)

        del target_hiddens
        torch.cuda.empty_cache()

        if iter_id % args.save_freq == 0:
            checkpoint = dict(last_batch_end=batch_end, data=data)
            save_checkpoint(checkpoint, args.output_path)

    checkpoint = dict(last_batch_end=batch_end, data=data)
    save_checkpoint(checkpoint, args.output_path)

    folder_path = os.path.dirname(args.output_path)
    done_file = os.path.join(folder_path, f"done_{args.process_id}.txt")
    with open(done_file, "w") as f:
        f.write("done\n")

    logger.info(f"Process {args.process_id} has finished. Created {done_file}")

    if args.process_id == 0:
        logger.info("Process 0 is waiting for all other processes to finish...")
        args.output_path = orig_output_path
        while True:
            done_files = [f"done_{i}.txt" for i in range(1, args.n_processes) if i * block_size < n]
            all_done = all(os.path.exists(os.path.join(folder_path, f)) for f in done_files) or args.n_processes == 1

            if all_done:
                logger.info("All processes have finished.")
                all_processes_data = []
                for i in range(args.n_processes):
                    process_output_file = f'{args.output_path}_{i}.pt'
                    checkpoint = torch.load(process_output_file)
                    all_processes_data.extend(checkpoint['data'])
                    assert np.all([len(j['changed_token_indices']) == len(j['target_hiddens']) for j in checkpoint['data']])
                    logger.info(f"Removing {process_output_file}")
                    os.remove(process_output_file)

                    #remove done file
                    done_file = os.path.join(folder_path, f"done_{i}.txt")
                    logger.info(f"Removing {done_file}")
                    os.remove(done_file)

                all_processes_data_path = f'{args.output_path}.pt'
                save_checkpoint(all_processes_data, all_processes_data_path)
                logger.info(f"Saved all processes data to {all_processes_data_path}")
                break

            print("Time Sleep")
            time.sleep(5)
