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
Preprocess the TACO dataset to verl parquet format.

This follows the critic-rl TACO split:
- function-call tasks are converted to Python assert tests
- standard-input tasks are converted to stdin/stdout tests
"""

import argparse
import json
import os
import re
import sys

import datasets

from verl.utils.hdfs_io import copy, makedirs

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000)


def _parse_input_output(raw_input_output):
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
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or not outputs:
        raise ValueError("input_output must contain non-empty inputs/outputs lists")
    if len(inputs) != len(outputs):
        raise ValueError(f"Mismatched test cases: {len(inputs)=}, {len(outputs)=}")
    return parsed


def _normalize_taco_io_value(inputs, outputs):
    """Match TACO's own lightweight normalization from critic-rl."""
    try:
        if isinstance(inputs[0], dict):
            inputs = [{int(k): v for k, v in inputs[0].items()}]
    except Exception:
        pass

    try:
        if isinstance(outputs, dict):
            outputs = [{int(k): v for k, v in outputs.items()}]
    except Exception:
        pass

    try:
        if isinstance(outputs[0], dict):
            outputs = [{int(k): v for k, v in outputs[0].items()}]
    except Exception:
        pass

    if isinstance(outputs, list) and outputs:
        outputs = outputs[0]

    return inputs, outputs


def _create_function_call_str(func_name, args_list):
    args_str = ", ".join(repr(arg) for arg in args_list)
    return f"{func_name}({args_str})"


def _build_tests(raw_input_output):
    parsed = _parse_input_output(raw_input_output)
    fn_name = parsed.get("fn_name") or None
    tests = []

    for raw_inputs, raw_outputs in zip(parsed["inputs"], parsed["outputs"], strict=True):
        inputs, outputs = _normalize_taco_io_value(raw_inputs, raw_outputs)

        if fn_name is None:
            if isinstance(inputs, list) or isinstance(outputs, list):
                continue
            if not isinstance(inputs, str) or not inputs.strip():
                continue
            if " = " in inputs or "[]" in inputs:
                continue
            tests.append(
                {
                    "input": {"stdin": inputs},
                    "output": {"stdout": str(outputs)},
                }
            )
        else:
            if isinstance(inputs, str):
                continue
            tests.append(
                f"assert {_create_function_call_str(fn_name, inputs)} == {repr(outputs)}".replace("'\"", '"').replace(
                    "\"'", '"'
                )
            )

    if not tests:
        raise ValueError("no compatible tests")

    return ("assert" if fn_name else "io"), tests, fn_name


def _extract_time_limit(raw_time_limit):
    if raw_time_limit is None:
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(raw_time_limit))
    return float(match.group()) if match else None


def _build_prompt(question, starter_code, test_type, tests, fn_name):
    prompt = (
        "You will be given a programming problem. Write a correct Python solution that passes all tests.\n\n"
        f"Problem:\n{question.strip()}\n\n"
    )

    if test_type == "assert":
        prompt += f"The judge will call `{fn_name}` directly. "
        if starter_code and starter_code.strip():
            prompt += (
                "Use the following starter code if needed, and return the final solution inside a Python code block.\n"
                f"```python\n{starter_code.strip()}\n```\n"
            )
        else:
            prompt += (
                "Return the final solution inside a Python code block.\n"
                "```python\n# YOUR CODE HERE\n```\n"
            )
    else:
        prompt += "Read from standard input and write to standard output. "
        if starter_code and starter_code.strip():
            prompt += (
                "Use the following starter code if needed, and return the final solution inside a Python code block.\n"
                f"```python\n{starter_code.strip()}\n```\n"
            )
        else:
            prompt += (
                "Return the final solution inside a Python code block.\n"
                "```python\n# YOUR CODE HERE\n```\n"
            )
    return prompt


def make_map_fn(split):
    def process_fn(example, idx):
        question = example.get("question", "").strip()
        if not question:
            raise ValueError("question is empty")
        if "<image>" in question or "<span " in question:
            raise ValueError("question contains unsupported markup")

        test_type, tests, fn_name = _build_tests(example.get("input_output"))
        starter_code = example.get("starter_code", "") or ""
        time_limit = _extract_time_limit(example.get("time_limit"))
        ground_truth = json.dumps(
            {
                "type": test_type,
                "tests": tests,
                "fn_name": fn_name,
                "time_limit": time_limit,
            },
            ensure_ascii=False,
        )

        return {
            "data_source": "taco",
            "prompt": [{"role": "user", "content": _build_prompt(question, starter_code, test_type, tests, fn_name)}],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "split": split,
                "index": idx,
                "task_id": example.get("task_id"),
                "difficulty": example.get("difficulty"),
                "source": example.get("source"),
                "tags": example.get("tags"),
                "time_limit": time_limit,
                "test_type": test_type,
                "fn_name": fn_name,
                "starter_code": starter_code,
                "question": question,
            },
        }

    return process_fn


def add_valid_flag(example):
    try:
        question = example.get("question", "")
        if "<image>" in question or "<span " in question:
            raise ValueError("question contains unsupported markup")
        _build_tests(example.get("input_output"))
        return {"_valid": True}
    except Exception:
        return {"_valid": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="Local dataset path if it already exists.")
    parser.add_argument("--dataset_name", default="BAAI/TACO", help="Hugging Face dataset name.")
    parser.add_argument("--local_save_dir", default="~/data/taco", help="Directory to save processed parquet files.")
    parser.add_argument(
        "--remove_columns",
        action="store_true",
        help="Remove original columns after preprocessing to reduce parquet size.",
    )
    args = parser.parse_args()

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, trust_remote_code=True)
    else:
        dataset = datasets.load_dataset(args.dataset_name, trust_remote_code=True)

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

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
