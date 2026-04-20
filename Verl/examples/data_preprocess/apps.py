# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the APPS dataset to verl parquet format.
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def build_prompt(question: str, starter_code: str | None) -> str:
    prompt = (
        "You will be given a programming problem. Write a correct Python solution that passes all hidden tests.\n\n"
        f"Problem:\n{question.strip()}\n\n"
    )

    if starter_code and starter_code.strip():
        prompt += (
            "Use the following starter code if needed, and return the final solution inside a Python code block.\n"
            f"```python\n{starter_code.strip()}\n```\n"
        )
    else:
        prompt += (
            "Read from standard input and write to standard output unless the problem specifies a function signature.\n"
            "Return the final solution inside a Python code block.\n"
            "```python\n# YOUR CODE HERE\n```\n"
        )

    return prompt


def normalize_test_cases(raw_input_output):
    if raw_input_output is None:
        raise ValueError("input_output is None")

    if isinstance(raw_input_output, str):
        raw_input_output = raw_input_output.strip()
        if not raw_input_output:
            raise ValueError("input_output is empty")
        parsed = json.loads(raw_input_output)
    elif isinstance(raw_input_output, dict):
        parsed = raw_input_output
    else:
        raise TypeError(f"Unsupported input_output type: {type(raw_input_output)}")

    inputs = parsed.get("inputs")
    outputs = parsed.get("outputs")
    fn_name = parsed.get("fn_name", None)

    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) == 0 or len(outputs) == 0:
        raise ValueError("input_output must contain non-empty inputs/outputs lists")

    if len(inputs) != len(outputs):
        raise ValueError(f"Mismatched test cases: {len(inputs)=}, {len(outputs)=}")

    normalized = {
        "inputs": inputs,
        "outputs": outputs,
        "fn_name": fn_name,
    }
    return json.dumps(normalized, ensure_ascii=False)


def make_map_fn(split):
    def process_fn(example, idx):
        question = example.get("question", "").strip()
        if not question:
            raise ValueError("question is empty")

        ground_truth = normalize_test_cases(example.get("input_output"))
        starter_code = example.get("starter_code", "")

        return {
            "data_source": "apps",
            "prompt": [{"role": "user", "content": build_prompt(question, starter_code)}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "problem_id": example.get("problem_id"),
                "difficulty": example.get("difficulty"),
                "url": example.get("url"),
                "starter_code": starter_code,
                "question": question,
            },
        }

    return process_fn


def add_valid_flag(example):
    try:
        question = example.get("question", "")
        _ = build_prompt(question, example.get("starter_code", ""))
        _ = normalize_test_cases(example.get("input_output"))
        return {"_valid": True}
    except Exception:
        return {"_valid": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="Local dataset path if it already exists.")
    parser.add_argument("--dataset_name", default="codeparrot/apps", help="Hugging Face dataset name.")
    parser.add_argument("--local_save_dir", default="~/data/apps", help="Directory to save processed parquet files.")
    parser.add_argument(
        "--remove_columns",
        action="store_true",
        help="Remove original columns after preprocessing to reduce parquet size.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(dataset_name)

    train_dataset = dataset["train"]
    test_split_name = "test" if "test" in dataset else "validation"
    test_dataset = dataset[test_split_name]

    train_dataset = train_dataset.map(add_valid_flag, num_proc=8)
    test_dataset = test_dataset.map(add_valid_flag, num_proc=8)

    train_dataset = train_dataset.filter(lambda example: example["_valid"], num_proc=8)
    test_dataset = test_dataset.filter(lambda example: example["_valid"], num_proc=8)

    train_remove_columns = train_dataset.column_names if args.remove_columns else ["_valid"]
    test_remove_columns = test_dataset.column_names if args.remove_columns else ["_valid"]

    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        remove_columns=train_remove_columns,
        num_proc=8,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn(test_split_name),
        with_indices=True,
        remove_columns=test_remove_columns,
        num_proc=8,
    )


    local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
