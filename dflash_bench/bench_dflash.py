"""
DFlash Throughput + Acceptance Benchmark for SGLang on Ascend NPU.

Reads acceptance directly from /get_server_info (avg_spec_accept_length).
Falls back to log parsing if the field isn't found.
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


def find_in_nested(obj, target_key):
    """Recursively yield values under any 'target_key' in nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                yield v
            yield from find_in_nested(v, target_key)
    elif isinstance(obj, list):
        for item in obj:
            yield from find_in_nested(item, target_key)


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return [{"prompt": ex["question"], "id": i}
            for i, ex in enumerate(ds.select(range(min(n, len(ds)))))]


def load_math500(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"prompt": ex["problem"], "id": i}
            for i, ex in enumerate(ds.select(range(min(n, len(ds)))))]


def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return [{"prompt": ex["prompt"], "id": i}
            for i, ex in enumerate(ds.select(range(min(n, len(ds)))))]


DATASETS = {"gsm8k": load_gsm8k, "math500": load_math500, "humaneval": load_humaneval}


@dataclass
class SampleResult:
    sample_id: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None


async def get_acceptance(base_url: str) -> Optional[float]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base_url}/get_server_info")
            r.raise_for_status()
            data = r.json()
            for v in find_in_nested(data, "avg_spec_accept_length"):
                return v
    except Exception as e:
        print(f"WARN: could not read acceptance: {e}")
    return None


async def query_one(client, base_url, model, sample, max_new_tokens, temperature):
    payload = {"model": model, "prompt": sample["prompt"],
               "max_tokens": max_new_tokens, "temperature": temperature}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{base_url}/v1/completions", json=payload, timeout=300.0)
        r.raise_for_status()
        d = r.json()
        usage = d.get("usage", {})
        return SampleResult(
            sample_id=sample["id"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=time.perf_counter() - t0,
        )
    except Exception as e:
        return SampleResult(sample_id=sample["id"], latency_s=time.perf_counter() - t0,
                            error=str(e))


async def run_benchmark(args):
    print(f"=== DFlash Benchmark ===")
    print(f"Dataset:    {args.dataset}")
    print(f"N samples:  {args.num_samples}  (warmup: {args.warmup})")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temp:       {args.temperature}")
    print(f"Concurrency: {args.concurrency}")
    print()

    print(f"Loading {args.dataset}...")
    samples = DATASETS[args.dataset](args.num_samples + args.warmup)
    warmup_samples = samples[:args.warmup]
    bench_samples = samples[args.warmup:args.warmup + args.num_samples]

    # Snapshot acceptance before benchmark
    acc_before = await get_acceptance(args.base_url)
    print(f"avg_spec_accept_length BEFORE benchmark: {acc_before}")
    print()

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(args.concurrency)
        async def _go(s):
            async with sem:
                return await query_one(client, args.base_url, args.model, s,
                                       args.max_new_tokens, args.temperature)

        # Warmup
        if warmup_samples:
            print(f"--- Warmup ({len(warmup_samples)}) ---")
            await asyncio.gather(*[_go(s) for s in warmup_samples])

        # Measured run
        print(f"--- Benchmark ({len(bench_samples)}) ---")
        t_start = time.perf_counter()
        results = []
        for coro in asyncio.as_completed([_go(s) for s in bench_samples]):
            r = await coro
            status = "OK" if r.error is None else f"ERR({r.error[:40]})"
            print(f"  sample {r.sample_id:3d}: {r.completion_tokens:4d} toks  "
                  f"{r.latency_s:6.2f}s  {status}")
            results.append(r)
        wall = time.perf_counter() - t_start

    # Snapshot acceptance after benchmark
    acc_after = await get_acceptance(args.base_url)

    successes = [r for r in results if r.error is None]
    total_completion = sum(r.completion_tokens for r in successes)
    lats = [r.latency_s for r in successes]

    print()
    print("=== Results ===")
    print(f"Samples:          {len(successes)}/{len(results)}")
    print(f"Wall clock:       {wall:.2f} s")
    print(f"Total tokens:     {total_completion}")
    print()
    print(f"** Throughput:    {total_completion/wall:.2f} tokens/s **")
    print()
    if lats:
        print(f"Latency mean:     {statistics.mean(lats):.2f} s")
        print(f"Latency median:   {statistics.median(lats):.2f} s")
        print(f"Latency min/max:  {min(lats):.2f} / {max(lats):.2f} s")
    print()
    print(f"avg_spec_accept_length BEFORE: {acc_before}")
    print(f"avg_spec_accept_length AFTER:  {acc_after}")
    if acc_after is not None:
        print(f"** Acceptance:    {acc_after:.3f} tokens/step **")
        print(f"   Idealised speedup ceiling: {1+acc_after:.2f}x")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS.keys(), default="gsm8k")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", default="/share/canada_group_folder/ckpt/Qwen3-8B")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
