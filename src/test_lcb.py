# SelfJudge
# Copyright (c) 2026-present NAVER Cloud Corp.
# Apache-2.0
#
# Derived from garipovroma/autojudge (Apache-2.0):
#   https://github.com/garipovroma/autojudge/blob/cf180e89a8718e10a34ccf222a2a96e786e24a20/src/find_important_tokens_lcb.py
# Modifications for SelfJudge:
#   - the mining loop was removed; the script only generates target-model solutions
#     for a slice of LiveCodeBench (--start_idx / --n_tasks) and appends pass@1
#     scores to output/lcb_result.jsonl.

import argparse
import json
import os
import time

import numpy as np
import torch
import transformers

from livecodebench_v5 import extract_code, codegen_metrics, unpack_lcb_data, CodeGenerationProblem


prompt_with_8_shots = "Given the following problem, reason and give a final answer to the problem.\nProblem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.\n Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: "
prompt_with_0_shots = "Given the following problem, reason and give a final answer to the problem.\n"
formatting_prompt = "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."


def verify_args(args):
    assert args.process_id < args.world_size, "--process_id must be < --world_size"


def get_args():
    parser = argparse.ArgumentParser(description="Arguments for important tokens selection algorithm")

    parser.add_argument('--draft_model', type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help='Path or identifier of the draft model.')
    parser.add_argument('--target_model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Path or identifier of the target model.')
    parser.add_argument('--torch_dtype', type=str, choices=['float32', 'auto'], default='auto',
                        help='Data type for torch tensors.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--num_next_token', type=int, default=5,
                        help='Number of future token for Likelihood Difference.')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--output_folder', type=str, default='compare_lcb',
                        help='Output folder name.')
    parser.add_argument('--output_file', type=str, default='important_tokens',
                        help='Output file name.')
    parser.add_argument('--world_size', type=int, default=1,
                        help='world size')
    parser.add_argument('--process_id', type=int, default=0,
                        help='Process ID')
    parser.add_argument('--dump_freq', type=int, default=64,
                        help='Dump frequency.')
    parser.add_argument('--write_index_offset', type=int, default=0,
                        help='Write index offset.')
    parser.add_argument('--keep_responses', type=bool, default=False,
                        help='Keep all responses in find_important_tokens algorithm output.')
    parser.add_argument('--start_idx', type=int, default=45)
    parser.add_argument('--n_tasks', type=int, default=880)
    parser.add_argument('--num_process_evaluate', type=int, default=4)
    parser.add_argument('--process_saver_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--lcb_path', type=str, default='none')

    args = parser.parse_args()
    verify_args(args)

    return args

def test_program(program_tokens, eval_samples, tokenizer):
    code = extract_code(tokenizer.batch_decode(program_tokens)[0])
    result = codegen_metrics([eval_samples], [[code]], num_process_evaluate=args.num_process_evaluate)
    score = int(result[0]['pass@1'])
    return dict(score=score, code=code, gen=tokenizer.batch_decode(program_tokens)[0], result=result)

@torch.no_grad()
def find_important_tokens(
        prompt: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizer,
        draft_model: transformers.AutoModelForCausalLM,
        target_model: transformers.AutoModelForCausalLM,
        eval_samples: list,
        sample: dict,
        num_next_token = 5
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

    target_score = test_program(target_response, eval_samples, tokenizer)['score']

    return target_score


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
    tokenizer.pad_token_id = 128004  # <|finetune_right_pad_id|>
    draft_model.generation_config.pad_token_id = target_model.generation_config.pad_token_id = tokenizer.pad_token_id

    from datasets import load_dataset

    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5", trust_remote_code=True)

    dataset = dataset.select(range(args.start_idx, min(args.start_idx + args.n_tasks, len(dataset))))
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore

    livecodebench_v5_dataset, livecodebench_v5_inputs_outputs = unpack_lcb_data(dataset, tokenizer)

    n_samples = args.n_tasks
    shard_len = (n_samples + args.world_size - 1) // args.world_size
    shard_start = args.process_id * shard_len
    shard_end = min((args.process_id + 1) * shard_len, n_samples)

    mined_data = []
    sample_idx = 0
    for sample_idx in range(len(dataset)):
        id = sample_idx + args.start_idx
        task = dataset[sample_idx]
        task_dict = livecodebench_v5_dataset[sample_idx]
        eval_samples = livecodebench_v5_inputs_outputs[sample_idx]
        batch = tokenizer(task_dict['prompt'], add_special_tokens=False, return_tensors='pt').to(device)

        output = find_important_tokens(batch['input_ids'], tokenizer, draft_model, target_model, eval_samples, dataset[sample_idx],
                                            num_next_token=args.num_next_token)

        spath = 'output/lcb_result.jsonl'

        print("ID:", sample_idx, 'Target_score:', output)

        del task_dict['tests']

        with open(os.path.expanduser(spath), "a") as fout:
            ans_json = {
                "id": id,
                "score": output,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    print('Done')

    