import os

import sys
import json
import time
import signal
import subprocess
import argparse
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from openai import OpenAI
import requests

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

try:
    multiprocessing.set_start_method("forkserver")
except RuntimeError:
    pass

from verl.utils.reward_score.taco import compute_score as taco_compute_score
# ===================== 配置区 =====================

# HF 模型路径（用来算 embedding）
MODEL_NAME = "/opt/huawei/dataset/lcc_guiyang/code/o_model"
# MODEL_NAME = "/home/ma-user/work/dataset/lcc_test/taco_280"
# vLLM server 的 served model name（启动 server 时要一致）
VLLM_MODEL_NAME = "qwen2.5-7b-instruct-tuned"

VLLM_HOST = "127.0.0.1"
VLLM_PORT_BASE = 8000
VLLM_API_KEY = "EMPTY"  
#避免token多线程的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TENSOR_PARALLEL_SIZE = 1
# PIPLINE_PARALLEL_SIZE = 2
# 采样参数
N_SAMPLES = 16
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 64
DATA_PATH = "/opt/huawei/dataset/lcc_guiyang/code/taco/train.parquet"
# DATA_PATH = "/cache/data/taco"
# 输出数据
OUT_DIR = Path("/opt/huawei/dataset/lcc_guiyang/generate_data/8b_rl_code")
# OUT_DIR = Path("/home/ma-user/work/dataset/lcc_test/tmp_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# HF 算 embedding 的 batch size
EMB_BATCH_SIZE = 32

# vLLM 启动后等待的最长时间（秒）
SERVER_START_TIMEOUT = 360


# ===================== 工具函数 =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument(
        "--grade-workers",
        type=int,
        default=int(os.getenv("TACO_GRADE_WORKERS", "64")),
        help="Total concurrent TACO response graders.",
    )
    return parser.parse_args()
    
def _load_jsonl_file(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _load_parquet_file(path: Path) -> List[Dict]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Loading parquet requires pandas/pyarrow installed.") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def load_dataset(path: str) -> List[Dict]:
    """
    加载 path 对应的所有 JSONL 数据。
    - 如果 path 是目录：遍历目录下所有 *.jsonl 文件（按文件名排序）
    - 如果 path 是文件：只加载该文件
    每一行解析为一个 dict，追加到 data 列表中。
    """
    data: List[Dict] = []
    p = Path(path)
    if p.is_dir():
        data_files = sorted(p.glob("*.parquet")) + sorted(p.glob("*.jsonl"))
    elif p.is_file():
        data_files = [p]
    else:
        raise FileNotFoundError(f"[load_dataset] path does not exist: {p}")

    if not data_files:
        print(f"[load_dataset] Warning: no .parquet/.jsonl files found under {p}")

    print("[load_dataset] loading files:")
    for f in data_files:
        print("  -", f)

    for fpath in data_files:
        if fpath.suffix == ".parquet":
            data.extend(_load_parquet_file(fpath))
        elif fpath.suffix == ".jsonl":
            data.extend(_load_jsonl_file(fpath))
        else:
            raise ValueError(f"Unsupported data file type: {fpath}")

    print(f"[load_dataset] total samples: {len(data)}")
    return data
        


def build_code_prompt_text(tokenizer, example: Dict) -> str:
    prompt = example["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    raise TypeError(f"Unsupported TACO prompt type: {type(prompt)}")


def build_meta(example: Dict, prompt_text: str) -> Dict:
    reward_model = example.get("reward_model", {})
    extra_info = example.get("extra_info", {})
    ground_truth = reward_model.get("ground_truth")
    if ground_truth is None:
        raise ValueError("TACO parquet sample must contain reward_model.ground_truth")
    return {
        "question": extra_info.get("question", prompt_text),
        "answer": ground_truth,
        "reward_model": reward_model,
        "extra_info": extra_info,
        "data_source": example.get("data_source", "taco"),
    }


def grade_response(resp: str, ground_truth: str) -> tuple[bool, Dict]:
    score, metadata = taco_compute_score(resp, ground_truth, continuous=True)
    return score, {"score": float(score), "metadata": metadata}


def grade_indexed_response(prompt_idx: int, response_idx: int, resp: str, ground_truth: str):
    score, grade_info = grade_response(resp, ground_truth)
    return prompt_idx, response_idx, resp, score, grade_info


def iter_grade_tasks(questions_meta: List[Dict], responses_all: List[List[str]]):
    for idx, (meta, responses) in enumerate(zip(questions_meta, responses_all)):
        for response_idx, resp in enumerate(responses):
            yield idx, response_idx, resp, meta["answer"]


def compute_embeddings_hf(
    hf_model,
    tokenizer,
    prompt_texts: List[str],
    batch_size: int = 4,
) -> np.ndarray:
    """
    用 HF 模型批量计算 final-layer embedding。
    对每个样本，取最后一个有效 token 的 hidden state，返回 (N, D)。
    """
    device = next(hf_model.parameters()).device
    embeddings: List[np.ndarray] = []

    hf_model.eval()

    for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Computing embeddings (HF)"):
        batch_texts = prompt_texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = outputs.hidden_states[-1]  # (B, L, D)
            lengths = attention_mask.sum(dim=1) - 1  # 每个样本最后一个非 pad 的位置

            for b_idx, l in enumerate(lengths):
                h = last_hidden[b_idx, l, :]  # (D,)
                h_np = h.detach().cpu().float().numpy()
                embeddings.append(h_np)

    emb_arr = np.stack(embeddings, axis=0)
    return emb_arr


def wait_for_server(base_url: str, timeout: int = 120) -> None:
    """
    轮询 vLLM OpenAI server 是否就绪。
    - 如果在 timeout 内连上 /v1/models，说明 OK；
    - 如果子进程提前退出，则立刻读取 stdout/stderr 并抛出异常。
    """
    url = base_url.rstrip("/") + "/models"
    start = time.time()
    print(f"Waiting for vLLM server at {url} ...")

    while True:
        # 2) 尝试连 /v1/models
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print("vLLM server is ready.")
                return
            else:
                print(f"vLLM server not ready yet, HTTP {resp.status_code}")
        except Exception as e:
            # 典型就是 connection refused，这里只是打印一行
            print(f"Waiting for vLLM server... ({e})")

        if time.time() - start > timeout:
            raise RuntimeError("Timeout waiting for vLLM server to start.")

        time.sleep(20)

def get_available_device_count() -> int:
    """
    优先使用 NPU，其次 CUDA。
    如果有 ASCEND_RT_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES，则按其中限制。
    """
    # 优先看环境变量（很多集群会通过这个限制可见设备）
    env_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        devs = [d for d in env_devices.split(",") if d.strip() != ""]
        print(f"[get_available_device_count] From env visible devices: {devs}")
        return len(devs)

    # NPU
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                n = torch.npu.device_count()
                print(f"[get_available_device_count] NPU count: {n}")
                if n > 0:
                    return n
        except Exception:
            pass

    raise RuntimeError("No NPU/CUDA devices found, or they are not visible.")

def stop_vllm_server(proc: Optional[subprocess.Popen]) -> None:
    """
    停止 vLLM server。
    """
    if proc is None:
        return
    try:
        if proc.poll() is None:  # 还在运行
            print("Terminating vLLM server...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("vLLM server did not exit, killing...")
                proc.kill()
    except Exception as e:
        print("Error while stopping vLLM server:", e)


def start_vllm_server(
    port: int,
    served_model_name: str,
    visible_devices: List[int],
) -> subprocess.Popen:
    """
    用 subprocess 启动一个 vLLM OpenAI server。
    每个 server 使用一组 visible_devices。
    """
    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--host",
        VLLM_HOST,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--gpu-memory-utilization",
        "0.7",
    ]

    print(f"Starting vLLM server on port {port} with devices {visible_devices}, command:\n  " + " ".join(cmd))

    env = os.environ.copy()
    dev_str = ",".join(str(d) for d in visible_devices)
    # NPU / GPU 统一设置可见设备
    env["ASCEND_RT_VISIBLE_DEVICES"] = dev_str

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
        env=env,
    )
    return proc

def sample_with_openai_completion_batch(
    client: OpenAI,
    prompt_texts: List[str],
    n_samples: int,
    max_new_tokens: int,
    batch_size: int = 8,
    model_name: str = VLLM_MODEL_NAME,
) -> List[List[str]]:
    """
    使用 /v1/completions，batch 调用 vLLM。
    返回: responses_all[i] 是第 i 个样本的回答列表（长度为 n_samples）。
    """
    responses_all: List[List[str]] = [[] for _ in range(len(prompt_texts))]

    for start in tqdm(range(0, len(prompt_texts), batch_size), desc="Sampling via vLLM(OpenAI, batch)"):
        end = min(start + batch_size, len(prompt_texts))
        batch_prompts = prompt_texts[start:end]
        cur_batch_size = len(batch_prompts)

        resp = client.completions.create(
            model=model_name,
            prompt=batch_prompts,
            max_tokens=max_new_tokens,
            temperature=1.0,
            n=n_samples,
        )
        choices = resp.choices

        expected = cur_batch_size * n_samples
        if len(choices) != expected:
            raise RuntimeError(
                f"Unexpected number of choices: got {len(choices)}, "
                f"expected {expected} (= {cur_batch_size} * {n_samples})"
            )

        for i in range(cur_batch_size):
            answers_i = []
            for k in range(n_samples):
                idx_choice = i * n_samples + k
                answers_i.append(choices[idx_choice].text)
            responses_all[start + i] = answers_i

    return responses_all

def split_ranges(num_items: int, num_parts: int) -> List[tuple]:
    """
    将 [0, num_items) 均匀切成 num_parts 段，返回 [(s0,e0), (s1,e1), ...]
    """
    ranges = []
    base = num_items // num_parts
    rem = num_items % num_parts
    start = 0
    for i in range(num_parts):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges

def worker_sample(server_idx, start, end, base_url, model_name, prompt_texts_slice):
    print(f"\n[Server {server_idx}] Processing samples [{start}, {end}) "
          f"with base_url={base_url}, model={model_name}")
    client = OpenAI(
        base_url=base_url,
        api_key=VLLM_API_KEY,
        timeout=2400,
    )
    sub_responses = sample_with_openai_completion_batch(
        client=client,
        prompt_texts=prompt_texts_slice,
        n_samples=N_SAMPLES,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        model_name=model_name,
    )
    assert len(sub_responses) == (end - start)
    return server_idx, start, end, sub_responses
    
# ===================== 主流程 =====================
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vllm_procs: List[subprocess.Popen] = []  # ==== 修改：保存多个 server ====
    try:
        # ---------- 1) HF 模型：算 embedding ----------
        print("Loading tokenizer & HF model (for embeddings)...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        print("Loading dataset from", args.data_path)
        data = load_dataset(args.data_path)
        N = len(data)
        print("Total examples:", len(data))
        ranges = split_ranges(N, args.num_shards)
        my_start, my_end = ranges[args.shard_id]
        print(f"[Rank {args.shard_id}] handling samples [{my_start}, {my_end})")
        
        # 构造 prompt_texts 和 meta
        prompt_texts: List[str] = []
        questions_meta: List[Dict] = []

        for example in data:
            prompt_text = build_code_prompt_text(tokenizer, example)
            meta = build_meta(example, prompt_text)
            prompt_texts.append(prompt_text)
            questions_meta.append(meta)

        data = data[my_start:my_end]
        prompt_texts = prompt_texts[my_start:my_end]
        questions_meta = questions_meta[my_start:my_end]
        # HF 批量算 embedding
        emb_arr = compute_embeddings_hf(
            hf_model=hf_model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            batch_size=EMB_BATCH_SIZE,
        )
        print("Embeddings computed, shape:", emb_arr.shape)

        # 释放 HF 模型显存，为 vLLM 腾空间
        del hf_model
        torch.cuda.empty_cache()
        time.sleep(3)

        # ---------- 2) 计算 NPU / GPU 数，决定要启动多少个 vLLM server ----------
        num_devices = get_available_device_count()
        num_servers = max(1, num_devices // TENSOR_PARALLEL_SIZE)
        if num_servers == 0:
            raise RuntimeError(
                f"Device count ({num_devices}) smaller than TENSOR_PARALLEL_SIZE ({TENSOR_PARALLEL_SIZE})."
            )

        if num_devices % TENSOR_PARALLEL_SIZE != 0:
            print(
                f"[Warning] device_count={num_devices} 不能被 TP={TENSOR_PARALLEL_SIZE} 整除，"
                f"只使用前 {num_servers * TENSOR_PARALLEL_SIZE} 个设备。"
            )

        print(f"Will start {num_servers} vLLM servers, each with TP={TENSOR_PARALLEL_SIZE}.")

        device_ids = list(range(num_devices))
        device_groups: List[List[int]] = []
        for i in range(num_servers):
            group = device_ids[i * TENSOR_PARALLEL_SIZE: (i + 1) * TENSOR_PARALLEL_SIZE]
            device_groups.append(group)

        # ---------- 3) 启动多个 vLLM server ----------
        base_urls = []
        model_names = []
        for i in range(num_servers):
            port = VLLM_PORT_BASE + i
            served_name = f"{VLLM_MODEL_NAME}_s{i}"
            proc = start_vllm_server(port=port, served_model_name=served_name, visible_devices=device_groups[i])
            vllm_procs.append(proc)

            base_url = f"http://{VLLM_HOST}:{port}/v1"
            base_urls.append(base_url)
            model_names.append(served_name)

        # 等待所有服务就绪
        for url in base_urls:
            wait_for_server(url, timeout=SERVER_START_TIMEOUT)

        # ---------- 4) 构造全局 responses_all 容器 ----------
        N = len(prompt_texts)
        responses_all: List[List[str]] = [None] * N  # type: ignore

        # 将数据平均切给多个 server
        ranges = split_ranges(N, num_servers)
        print("Data split ranges for servers:", ranges)

        # ---------- 5) 多 server 采样 ----------
        tasks = []
        with ThreadPoolExecutor(max_workers=len(base_urls)) as executor:
            for server_idx, ((start, end), base_url, model_name) in enumerate(
                zip(ranges, base_urls, model_names)
            ):
                if start == end:
                    continue
                sub_prompts = prompt_texts[start:end]
                fut = executor.submit(
                    worker_sample,
                    server_idx,
                    start,
                    end,
                    base_url,
                    model_name,
                    sub_prompts,
                )
                tasks.append(fut)
        
            for fut in as_completed(tasks):
                server_idx, start, end, sub_responses = fut.result()
                for idx_local, resp_list in enumerate(sub_responses):
                    responses_all[start + idx_local] = resp_list

        # 安全检查
        assert all(r is not None for r in responses_all)

        # ---------- 6) 判分，统计 k ----------
        print("Grading responses...")
        k_list: List[float] = [0.0] * len(questions_meta)
        response_items: List[List[Optional[Dict]]] = [[None] * len(responses) for responses in responses_all]
        total_responses = sum(len(responses) for responses in responses_all)
        grade_workers = max(1, int(args.grade_workers))
        print(f"Grading {total_responses} responses with {grade_workers} workers.", flush=True)

        with ThreadPoolExecutor(max_workers=grade_workers) as executor:
            task_iter = iter(iter_grade_tasks(questions_meta, responses_all))
            pending = set()
            max_pending = max(grade_workers * 2, grade_workers)

            for _ in range(min(max_pending, total_responses)):
                try:
                    pending.add(executor.submit(grade_indexed_response, *next(task_iter)))
                except StopIteration:
                    break
            print(f"Submitted initial {len(pending)} grading futures.", flush=True)

            progress = tqdm(
                total=total_responses,
                desc="Grading responses",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx, response_idx, resp, score, grade_info = future.result()
                    k_list[idx] += score
                    resp_item = {"response": resp, "score": score}
                    resp_item.update(grade_info)
                    response_items[idx][response_idx] = resp_item
                    progress.update(1)

                    try:
                        pending.add(executor.submit(grade_indexed_response, *next(task_iter)))
                    except StopIteration:
                        pass
            progress.close()
                

        for idx, meta in enumerate(questions_meta):
            meta["k"] = float(k_list[idx])
            meta["responses"] = response_items[idx]
            if idx < 3:
                print("=" * 40)
                print(f"Example #{idx}")
                print("Q:", meta["question"])
                print("gold:", meta["answer"])
                print("k:", k_list[idx])

        k_arr = np.array(k_list, dtype=np.float32)
        print("k array shape:", k_arr.shape)
        
        # ---------- 7) 保存结果 ----------
        OUT_EMB_PATH = out_dir / f"probe_train_embeddings_rank{args.shard_id}.npy"
        OUT_K_PATH   = out_dir / f"probe_train_k_rank{args.shard_id}.npy"
        OUT_Q_PATH   = out_dir / f"result_rank{args.shard_id}.jsonl"        
        
        np.save(OUT_EMB_PATH, emb_arr)
        np.save(OUT_K_PATH, k_arr)

        with open(OUT_Q_PATH, "w", encoding="utf-8") as f:
            for item in questions_meta:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print("Saved embeddings to:", OUT_EMB_PATH)
        print("Saved k to:", OUT_K_PATH)
        print("Saved questions to:", OUT_Q_PATH)

    finally:
        # ---------- 8) 退出时把所有 vLLM server 停掉 ----------
        for proc in vllm_procs:
            stop_vllm_server(proc)


if __name__ == "__main__":
    main()




