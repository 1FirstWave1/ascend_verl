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
Local TACO code reward.

This mirrors critic-rl's TACO contract without SandboxFusion:
- assert tasks run generated code plus one Python assert per test
- io tasks run generated code as a script with mocked stdin and captured stdout
"""

import contextlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any
from unittest.mock import mock_open, patch

_MAX_PRINTED_ERRORS = int(os.getenv("TACO_REWARD_MAX_PRINTED_ERRORS", "1"))

_PRELUDE = """
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""


def _print_failure(metadata: dict, code: str | None = None, index: int | None = None):
    prefix = "[TACO reward error]"
    if index is not None:
        prefix += f" case={index}"
    print(prefix, metadata.get("error", "unknown error"), file=sys.stderr)
    traceback_text = metadata.get("traceback")
    if traceback_text:
        print(traceback_text, file=sys.stderr)
    if code is not None:
        print("[TACO reward code begin]", file=sys.stderr)
        print(code, file=sys.stderr)
        print("[TACO reward code end]", file=sys.stderr)


def _extract_code(completion: str) -> str:
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    if "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():
                    return rest
            return code
    return completion


def _normalize_stdout(text: Any) -> str:
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    return str(text).strip()


def _outputs_match(actual: str, expected: Any) -> bool:
    actual = _normalize_stdout(actual)
    expected = _normalize_stdout(expected)
    if actual == expected:
        return True

    actual_lines = [line.strip() for line in actual.splitlines() if line.strip()]
    expected_lines = [line.strip() for line in expected.splitlines() if line.strip()]
    if actual_lines == expected_lines:
        return True

    actual_tokens = actual.split()
    expected_tokens = expected.split()
    if actual_tokens == expected_tokens:
        return True

    try:
        if len(actual_tokens) == len(expected_tokens):
            return all(abs(float(a) - float(e)) <= 1e-6 for a, e in zip(actual_tokens, expected_tokens, strict=True))
    except Exception:
        pass
    return False


def _run_assert_case(code: str, assert_case: str):
    namespace = {"__name__": "__main__"}
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        exec(_PRELUDE + "\n" + code + "\n" + assert_case, namespace, namespace)


def _run_io_case(code: str, test_case: dict):
    stdin = str(test_case.get("input", {}).get("stdin", ""))
    expected = test_case.get("output", {}).get("stdout", "")
    namespace = {"__name__": "__main__"}
    stdin_stream = io.StringIO(stdin)
    stdout_stream = io.StringIO()
    with (
        patch("builtins.open", mock_open(read_data=stdin)),
        patch("sys.stdin", stdin_stream),
        contextlib.redirect_stdout(stdout_stream),
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stderr(devnull),
    ):
        exec(_PRELUDE + "\n" + code, namespace, namespace)
    if not _outputs_match(stdout_stream.getvalue(), expected):
        raise AssertionError({"actual": stdout_stream.getvalue(), "expected": expected})


def _write_result(path: str, passed: bool, metadata: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "metadata": metadata}, f, ensure_ascii=False)


def _case_worker(code: str, test_type: str, test_case: Any, workdir: str, result_path: str):
    original_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        if test_type == "assert":
            _run_assert_case(code, test_case)
        elif test_type == "io":
            _run_io_case(code, test_case)
        else:
            raise ValueError(f"Unknown TACO test type: {test_type}")
        _write_result(result_path, True, {"passed": True})
    except Exception as exc:
        metadata = {
            "passed": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=5),
        }
        _write_result(
            result_path,
            False,
            metadata,
        )
    finally:
        try:
            os.chdir(original_cwd)
        except Exception:
            pass


def _run_case(code: str, test_type: str, test_case: Any, timeout: float):
    tmpdir = tempfile.mkdtemp(prefix="verl_taco_reward_")
    result_path = os.path.join(tmpdir, "result.json")
    try:
        proc = multiprocessing.Process(target=_case_worker, args=(code, test_type, test_case, tmpdir, result_path))
        proc.start()
        proc.join(timeout=max(float(timeout), 1.0))
        if proc.is_alive():
            proc.kill()
            proc.join()
            return False, {"passed": False, "error": "timeout"}
        if not os.path.exists(result_path):
            return False, {"passed": False, "error": "no result"}
        with open(result_path, encoding="utf-8") as f:
            payload = json.load(f)
        return bool(payload.get("passed")), dict(payload.get("metadata", {}))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compute_score(completion, test_cases, continuous=False):
    try:
        if not isinstance(test_cases, dict):
            test_cases = json.loads(test_cases)
        test_type = test_cases["type"]
        tests = test_cases["tests"]
        timeout = test_cases.get("time_limit") or 10
        code = _extract_code(completion)

        if continuous:
            tests = tests[:10]

        results = []
        metadata = []
        printed_errors = 0
        for case_index, test_case in enumerate(tests):
            passed, info = _run_case(code, test_type, test_case, timeout)
            results.append(passed)
            metadata.append(info)
            if not passed and printed_errors < _MAX_PRINTED_ERRORS:
                # _print_failure(info, code=code, index=case_index)
                printed_errors += 1

        score = sum(results) / len(results) if results else 0.0
        return float(score), metadata
    except Exception as exc:
        metadata = {"passed": False, "error": repr(exc), "traceback": traceback.format_exc(limit=5)}
        # _print_failure(metadata, code=_extract_code(completion) if isinstance(completion, str) else None)
        return 0.0, [metadata]
