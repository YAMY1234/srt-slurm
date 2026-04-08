#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K benchmark using OpenAI-compatible completions API.

Replicates sglang/benchmark/gsm8k/bench_sglang.py but uses the standard
OpenAI /v1/completions endpoint instead of sglang's native frontend language.
This ensures compatibility with disaggregated deployments behind nginx/proxy
and avoids unintended chat template injection.

Supports two modes:
  - Default: /v1/completions (raw text, no chat template)
  - --use-chat-api: /v1/chat/completions (server applies chat template)
"""

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

INVALID = -9999999
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)


# ---------------------------------------------------------------------------
# Data helpers (same logic as the original bench_sglang.py)
# ---------------------------------------------------------------------------

def read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_one_example(lines, i, include_answer):
    ret = f"Question: {lines[i]['question']}\nAnswer:"
    if include_answer:
        ret += f" {lines[i]['answer']}"
    return ret


def get_few_shot_examples(lines, k):
    return "".join(get_one_example(lines, i, True) + "\n\n" for i in range(k))


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def download_data(url: str) -> str:
    """Download GSM8K data, using sglang's cache utility if available."""
    try:
        from sglang.utils import download_and_cache_file
        return download_and_cache_file(url)
    except ImportError:
        import urllib.request
        import tempfile
        cache_dir = Path(tempfile.gettempdir()) / "gsm8k_cache"
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / "test.jsonl"
        if not path.exists():
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, path)
        return str(path)


# ---------------------------------------------------------------------------
# Async request helpers
# ---------------------------------------------------------------------------

async def _request_completions(session, endpoint, model, prompt,
                               max_tokens, temperature, top_p, stop, sem):
    """POST /v1/completions  (raw text, no chat template)."""
    url = f"{endpoint}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    async with sem:
        for attempt in range(7):
            try:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "text": data["choices"][0]["text"],
                            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        }
                    body = await resp.text()
                    print(f"[attempt {attempt}] HTTP {resp.status}: {body[:200]}", file=sys.stderr)
            except Exception as e:
                print(f"[attempt {attempt}] error: {e}", file=sys.stderr)
            await asyncio.sleep(min(2 ** attempt, 32))
    return {"text": "", "completion_tokens": 0}


async def _request_chat(session, endpoint, model, prompt,
                        max_tokens, temperature, top_p, stop, sem):
    """POST /v1/chat/completions  (server applies chat template)."""
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    async with sem:
        for attempt in range(7):
            try:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "text": data["choices"][0]["message"]["content"],
                            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        }
                    body = await resp.text()
                    print(f"[attempt {attempt}] HTTP {resp.status}: {body[:200]}", file=sys.stderr)
            except Exception as e:
                print(f"[attempt {attempt}] error: {e}", file=sys.stderr)
            await asyncio.sleep(min(2 ** attempt, 32))
    return {"text": "", "completion_tokens": 0}


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    # ---- load data --------------------------------------------------------
    if args.platinum:
        from datasets import load_dataset
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [{"question": item["question"], "answer": item["answer"]}
                 for item in dataset]
    else:
        data_path = args.data_path
        if not os.path.isfile(data_path):
            data_path = download_data(GSM8K_URL)
        lines = list(read_jsonl(data_path))

    num_shots = args.num_shots
    few_shot = get_few_shot_examples(lines, num_shots)

    questions, labels = [], []
    for i in range(min(args.num_questions, len(lines))):
        questions.append(few_shot + get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(la != INVALID for la in labels), "Some ground-truth labels could not be parsed"

    stop_words = ["Question", "Assistant:", "<|separator|>"]

    print(f"GSM8K-Bench config:")
    print(f"  endpoint     = {args.endpoint}")
    print(f"  model        = {args.model}")
    print(f"  num_questions= {len(questions)}")
    print(f"  num_shots    = {num_shots}")
    print(f"  max_new_tokens = {args.max_new_tokens}")
    print(f"  parallel     = {args.parallel}")
    print(f"  temperature  = {args.temperature}")
    print(f"  top_p        = {args.top_p}")
    print(f"  use_chat_api = {args.use_chat_api}")
    print(f"  platinum     = {args.platinum}")
    print()

    # ---- fire requests ----------------------------------------------------
    sem = asyncio.Semaphore(args.parallel)
    req_fn = _request_chat if args.use_chat_api else _request_completions

    connector = aiohttp.TCPConnector(limit=args.parallel + 16)
    async with aiohttp.ClientSession(connector=connector) as session:
        tic = time.perf_counter()
        tasks = [
            req_fn(session, args.endpoint, args.model, q,
                   args.max_new_tokens, args.temperature, args.top_p,
                   stop_words, sem)
            for q in questions
        ]
        results = await asyncio.gather(*tasks)
        latency = time.perf_counter() - tic

    # ---- evaluate ---------------------------------------------------------
    preds = [get_answer_value(r["text"]) for r in results]
    total_tokens = sum(r["completion_tokens"] for r in results)

    acc = float(np.mean(np.array(preds) == np.array(labels)))
    invalid_rate = float(np.mean(np.array(preds) == INVALID))
    throughput = total_tokens / latency if latency > 0 else 0.0

    print("=" * 60)
    print(f"Accuracy:          {acc:.4f}")
    print(f"Invalid rate:      {invalid_rate:.4f}")
    print(f"Latency:           {latency:.3f} s")
    print(f"Output tokens:     {total_tokens}")
    print(f"Output throughput: {throughput:.3f} token/s")
    print("=" * 60)

    task_name = "gsm8k_platinum" if args.platinum else "gsm8k"
    label_json = json.dumps({"model": args.model, "eval": f"{task_name}_bench"})
    print(f"[METRIC] {task_name}_bench_accuracy={acc} labels={label_json}")
    print(f"[METRIC] {task_name}_bench_latency={latency} labels={label_json}")
    print(f"[METRIC] {task_name}_bench_throughput={throughput} labels={label_json}")

    # ---- save results -----------------------------------------------------
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace("/", "_")
    result_file = result_dir / f"gsm8k_bench_{model_safe}.json"
    with open(result_file, "w") as f:
        json.dump({
            "task": task_name + "_bench",
            "accuracy": round(acc, 4),
            "invalid_rate": round(invalid_rate, 4),
            "latency": round(latency, 3),
            "output_tokens": total_tokens,
            "output_throughput": round(throughput, 3),
            "config": {
                "num_questions": len(questions),
                "num_shots": num_shots,
                "parallel": args.parallel,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "model": args.model,
                "use_chat_api": args.use_chat_api,
            },
        }, f, indent=2)
    print(f"Results saved to: {result_file}")


def main():
    p = argparse.ArgumentParser(description="GSM8K benchmark via OpenAI completions API")
    p.add_argument("--endpoint", required=True, help="Server base URL, e.g. http://localhost:8790")
    p.add_argument("--model", required=True, help="Served model name")
    p.add_argument("--num-shots", type=int, default=5)
    p.add_argument("--num-questions", type=int, default=1319)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--parallel", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--data-path", type=str, default="test.jsonl")
    p.add_argument("--result-dir", type=str, default="/logs/accuracy")
    p.add_argument("--use-chat-api", action="store_true",
                   help="Use /v1/chat/completions (applies chat template)")
    p.add_argument("--platinum", action="store_true",
                   help="Use GSM8K Platinum dataset (corrected labels)")
    args = p.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
