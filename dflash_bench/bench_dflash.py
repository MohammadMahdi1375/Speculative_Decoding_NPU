"""
DFlash Throughput Benchmark for SGLang on Ascend NPU.

Sends N samples from a chosen dataset to a running SGLang server and
reports throughput, latency, and (when available) speculative decoding
acceptance metrics.

Usage:
    python bench_dflash.py --dataset gsm8k --num-samples 20
    python bench_dflash.py --dataset humaneval --num-samples 50 --max-new-tokens 512
    python bench_dflash.py --dataset math500 --num-samples 10 --concurrency 4
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


# ----- Dataset loading -----

def load_gsm8k(n: int) -> list[dict]:
    """GSM8K test split — math word problems."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return [
        {"prompt": ex["question"], "reference": ex["answer"], "id": i}
        for i, ex in enumerate(ds.select(range(min(n, len(ds)))))
    ]


def load_math500(n: int) -> list[dict]:
    """MATH-500 — competition math problems."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {"prompt": ex["problem"], "reference": ex.get("solution", ""), "id": i}
        for i, ex in enumerate(ds.select(range(min(n, len(ds)))))
    ]


def load_humaneval(n: int) -> list[dict]:
    """HumanEval — Python code completion."""
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return [
        {"prompt": ex["prompt"], "reference": ex.get("canonical_solution", ""), "id": i}
        for i, ex in enumerate(ds.select(range(min(n, len(ds)))))
    ]


DATASETS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "humaneval": load_humaneval,
}


# ----- Result tracking -----

@dataclass
class SampleResult:
    sample_id: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    first_token_latency_s: Optional[float] = None
    output_text: str = ""
    error: Optional[str] = None


@dataclass
class AggregateResult:
    n_samples: int
    n_success: int
    n_failed: int
    wall_clock_s: float
    total_completion_tokens: int
    total_prompt_tokens: int
    per_sample_latency: list[float] = field(default_factory=list)
    server_info: dict = field(default_factory=dict)

    @property
    def throughput_tok_per_s(self) -> float:
        return self.total_completion_tokens / self.wall_clock_s if self.wall_clock_s else 0

    @property
    def avg_latency_s(self) -> float:
        return statistics.mean(self.per_sample_latency) if self.per_sample_latency else 0

    @property
    def median_latency_s(self) -> float:
        return statistics.median(self.per_sample_latency) if self.per_sample_latency else 0


# ----- Server interaction -----

async def query_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    sample: dict,
    max_new_tokens: int,
    temperature: float,
) -> SampleResult:
    payload = {
        "model": model,
        "prompt": sample["prompt"],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=300.0,
        )
        r.raise_for_status()
        data = r.json()
        elapsed = time.perf_counter() - t0

        usage = data.get("usage", {})
        text = data["choices"][0]["text"]
        return SampleResult(
            sample_id=sample["id"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_s=elapsed,
            output_text=text,
        )
    except Exception as e:
        return SampleResult(
            sample_id=sample["id"],
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_s=time.perf_counter() - t0,
            error=str(e),
        )


async def fetch_server_info(base_url: str) -> dict:
    """Try several SGLang endpoints to gather DFlash-relevant info."""
    info = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in ("/get_server_info", "/model_info"):
            try:
                r = await client.get(f"{base_url}{endpoint}")
                if r.status_code == 200:
                    info[endpoint] = r.json()
            except Exception:
                pass
    return info


# ----- Benchmark driver -----

async def run_benchmark(
    dataset_name: str,
    n_samples: int,
    base_url: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
    concurrency: int,
    warmup: int,
) -> tuple[AggregateResult, list[SampleResult]]:
    print(f"\n=== DFlash Benchmark ===")
    print(f"Dataset:        {dataset_name}")
    print(f"N samples:      {n_samples}")
    print(f"Model:          {model}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature:    {temperature}")
    print(f"Concurrency:    {concurrency}")
    print(f"Warmup:         {warmup}")
    print(f"Server:         {base_url}")
    print()

    # Load dataset (with warmup overhead)
    print(f"Loading {dataset_name}...")
    loader = DATASETS[dataset_name]
    samples = loader(n_samples + warmup)
    if len(samples) < n_samples + warmup:
        print(f"WARN: only got {len(samples)} samples, asked for {n_samples + warmup}")

    warmup_samples = samples[:warmup]
    bench_samples = samples[warmup:warmup + n_samples]

    server_info = await fetch_server_info(base_url)
    if server_info:
        print(f"Server info collected: {list(server_info.keys())}")

    # Warmup phase
    if warmup_samples:
        print(f"\n--- Warmup ({len(warmup_samples)} samples) ---")
        async with httpx.AsyncClient() as client:
            sem = asyncio.Semaphore(concurrency)
            async def _go(s):
                async with sem:
                    return await query_one(client, base_url, model, s, max_new_tokens, temperature)
            await asyncio.gather(*[_go(s) for s in warmup_samples])
        print("Warmup done.")

    # Measured phase
    print(f"\n--- Benchmark ({len(bench_samples)} samples, concurrency={concurrency}) ---")
    results: list[SampleResult] = []
    t_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)
        async def _go(s):
            async with sem:
                res = await query_one(client, base_url, model, s, max_new_tokens, temperature)
                status = "OK" if res.error is None else f"ERR({res.error[:50]})"
                print(f"  sample {s['id']:3d}: {res.completion_tokens:4d} toks  "
                      f"{res.latency_s:6.2f}s  {status}")
                return res
        results = await asyncio.gather(*[_go(s) for s in bench_samples])

    t_end = time.perf_counter()
    wall_clock = t_end - t_start

    # Aggregate
    successes = [r for r in results if r.error is None]
    agg = AggregateResult(
        n_samples=len(bench_samples),
        n_success=len(successes),
        n_failed=len(results) - len(successes),
        wall_clock_s=wall_clock,
        total_completion_tokens=sum(r.completion_tokens for r in successes),
        total_prompt_tokens=sum(r.prompt_tokens for r in successes),
        per_sample_latency=[r.latency_s for r in successes],
        server_info=server_info,
    )
    return agg, results


def print_report(agg: AggregateResult, results: list[SampleResult]):
    print("\n=== Results ===")
    print(f"Samples completed:        {agg.n_success}/{agg.n_samples}")
    if agg.n_failed:
        print(f"Failures:                 {agg.n_failed}")
    print(f"Wall clock:               {agg.wall_clock_s:.2f} s")
    print(f"Total completion tokens:  {agg.total_completion_tokens}")
    print(f"Total prompt tokens:      {agg.total_prompt_tokens}")
    print()
    print(f"** Throughput:            {agg.throughput_tok_per_s:.2f} tokens/s **")
    print()
    if agg.per_sample_latency:
        print(f"Latency mean:             {agg.avg_latency_s:.2f} s")
        print(f"Latency median:           {agg.median_latency_s:.2f} s")
        print(f"Latency min:              {min(agg.per_sample_latency):.2f} s")
        print(f"Latency max:              {max(agg.per_sample_latency):.2f} s")
        if len(agg.per_sample_latency) >= 5:
            sorted_lats = sorted(agg.per_sample_latency)
            p95_idx = int(len(sorted_lats) * 0.95)
            print(f"Latency p95:              {sorted_lats[p95_idx]:.2f} s")
    print()
    avg_completion = (agg.total_completion_tokens / agg.n_success) if agg.n_success else 0
    print(f"Avg completion length:    {avg_completion:.1f} tokens")
    print()
    print("Note: SGLang server-side log shows DFlash acceptance:")
    print("      grep 'DFLASH verify completed' /tmp/dflash_npu_v4.log | tail -20")
    print("      Count 'num_accepted_drafts_per_req' values for acceptance rate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS.keys(), default="gsm8k")
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", default="/share/canada_group_folder/ckpt/Qwen3-8B")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Concurrent requests. 1=sequential (clean per-sample timing).")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Number of warmup samples (not counted in measurement).")
    ap.add_argument("--save-results", type=str, default=None,
                    help="Optional: save per-sample results to this JSON file.")
    args = ap.parse_args()

    agg, results = asyncio.run(run_benchmark(
        dataset_name=args.dataset,
        n_samples=args.num_samples,
        base_url=args.base_url,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        warmup=args.warmup,
    ))
    print_report(agg, results)

    if args.save_results:
        out = {
            "dataset": args.dataset,
            "n_samples": agg.n_samples,
            "wall_clock_s": agg.wall_clock_s,
            "throughput_tok_per_s": agg.throughput_tok_per_s,
            "avg_latency_s": agg.avg_latency_s,
            "median_latency_s": agg.median_latency_s,
            "results": [
                {
                    "id": r.sample_id,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "latency_s": r.latency_s,
                    "error": r.error,
                    "output": r.output_text[:200],
                }
                for r in results
            ],
        }
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results → {args.save_results}")


if __name__ == "__main__":
    main()
