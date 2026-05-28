"""
DFlash benchmark — reads timing via HTTP /server_info, no log file needed.
"""
import argparse
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset


DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "split": "test",
        "format": lambda x: (
            f"{x['question']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "split": "test",
        "format": lambda x: (
            f"{x['problem']}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "split": "test",
        "format": lambda x: (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{x['prompt']}\n```"
        ),
    },
}


def load_data(name, n):
    cfg = DATASETS[name]
    ds = load_dataset(*cfg["load_args"], split=cfg["split"])
    items = [{"text": cfg["format"](x)} for x in ds]
    random.seed(42)
    random.shuffle(items)
    return items if n is None else items[:n]


def send_one(base_url, prompt, max_new_tokens, temperature, top_p, top_k, timeout):
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def get_spec_timings(base_url):
    """Returns (algo_name, timings_dict) from /server_info, or (None, None).
    Checks for dflash_timings or eagle_timings, whichever is present."""
    try:
        r = requests.get(f"{base_url}/server_info", timeout=10)
        r.raise_for_status()
        d = r.json()
        for state in d.get("internal_states", []):
            for algo in ("dflash", "eagle"):
                t = state.get(f"{algo}_timings")
                if t:
                    return algo, t
        return None, None
    except Exception as e:
        print(f"WARN: could not fetch spec timings: {e}")
        return None, None


def get_dflash_timings(base_url):
    """Backward-compat wrapper."""
    _, t = get_spec_timings(base_url)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="If omitted, benchmarks the entire dataset.")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--timeout-s", type=int, default=3600)
    args = ap.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.num_prompts is None:
        print(f"Loading FULL {args.dataset} dataset (plus {args.concurrency} warmup)...")
        raw = load_data(args.dataset, None)
    else:
        print(f"Loading {args.num_prompts} samples from {args.dataset} "
              f"(plus {args.concurrency} warmup)...")
        raw = load_data(args.dataset, args.num_prompts + args.concurrency)
    print(f"  Loaded {len(raw)} samples")

    prompts = []
    for item in raw:
        messages = [{"role": "user", "content": item["text"]}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompts.append(chat_text)

    # Snapshot timings BEFORE
    algo_before, t_before = get_spec_timings(args.base_url)
    if t_before:
        _parts = [f"step={t_before['step_count']}",
                  f"draft={t_before.get('pure_draft_total_s', 0.0):.2f}s",
                  f"verify={t_before.get('pure_verify_total_s', 0.0):.2f}s"]
        if 'other_total_s' in t_before:
            _parts.append(f"other={t_before['other_total_s']:.2f}s")
        print(f"Timings BEFORE: {' '.join(_parts)}")

    try:
        requests.get(f"{args.base_url}/flush_cache", timeout=60).raise_for_status()
    except Exception as e:
        print(f"WARN: flush_cache failed: {e}")

    bs = max(args.concurrency, 1)
    if len(prompts) > bs:
        print(f"Warmup ({bs} samples)...")
        with ThreadPoolExecutor(max_workers=bs) as pool:
            list(pool.map(
                lambda p: send_one(args.base_url, p, args.max_new_tokens,
                                   args.temperature, args.top_p, args.top_k,
                                   args.timeout_s),
                prompts[:bs]
            ))
        prompts = prompts[bs:]

    print(f"Benchmarking {len(prompts)} prompts, concurrency={args.concurrency}...")
    start = time.perf_counter()
    total_tokens = 0
    accept_lengths = []
    accept_rates = []
    accepted_drafts_sum = 0
    proposed_drafts_sum = 0
    verify_ct_sum = 0
    e2e_latencies = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(
            send_one, args.base_url, p, args.max_new_tokens,
            args.temperature, args.top_p, args.top_k, args.timeout_s,
        ): i for i, p in enumerate(prompts)}

        for fut in tqdm(as_completed(futures), total=len(prompts)):
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            accepted_drafts_sum += int(meta.get("spec_accepted_drafts", 0))
            proposed_drafts_sum += int(meta.get("spec_proposed_drafts", 0))
            if "e2e_latency" in meta:
                try: e2e_latencies.append(float(meta["e2e_latency"]))
                except (TypeError, ValueError): pass
            if "spec_accept_length" in meta:
                try: accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError): pass
            if "spec_accept_rate" in meta:
                try: accept_rates.append(float(meta["spec_accept_rate"]))
                except (TypeError, ValueError): pass

    elapsed = time.perf_counter() - start
    algo_after, t_after = get_spec_timings(args.base_url)

    print()
    print("=" * 60)
    print(f"Dataset:               {args.dataset}")
    print(f"Prompts:               {len(prompts)}")
    print(f"Concurrency:           {args.concurrency}")
    print(f"Wall clock:            {elapsed:.2f} s")
    print(f"Total output tokens:   {total_tokens}")
    print()
    print(f"** Throughput:         {total_tokens/elapsed:.2f} tok/s **")
    if accept_lengths:
        print(f"** Accept length:      {statistics.mean(accept_lengths):.3f} (mean) **")
        print(f"   Min / Max:          {min(accept_lengths):.2f} / {max(accept_lengths):.2f}")
        if len(accept_lengths) >= 2:
            print(f"   Stdev:              {statistics.stdev(accept_lengths):.3f}")
    if accept_rates:
        print(f"   Accept rate (mean): {100*statistics.mean(accept_rates):.2f}%")
    if proposed_drafts_sum > 0:
        print(f"   Overall accept rate: {100*accepted_drafts_sum/proposed_drafts_sum:.2f}%  "
              f"({accepted_drafts_sum}/{proposed_drafts_sum} drafts)")
    if verify_ct_sum > 0:
        print(f"   Verify ct sum:      {verify_ct_sum}")
    if e2e_latencies:
        print(f"   E2E latency (mean): {statistics.mean(e2e_latencies):.2f} s")
        print(f"   E2E latency (min):  {min(e2e_latencies):.2f} s")
        print(f"   E2E latency (max):  {max(e2e_latencies):.2f} s")
    print()

    # Timing breakdown via HTTP
    if t_before and t_after:
        d_step = t_after["step_count"] - t_before["step_count"]
        d_pd = t_after["pure_draft_total_s"] - t_before["pure_draft_total_s"]
        d_pv = t_after["pure_verify_total_s"] - t_before["pure_verify_total_s"]
        d_oth = t_after.get("other_total_s", 0.0) - t_before.get("other_total_s", 0.0)
        if d_step > 0:
            print(f"=== Timing Breakdown — {algo_after or algo_before or 'spec'} (delta over this run) ===")
            print(f"   Verify steps:       {d_step}")
            print(f"   Pure draft total:   {d_pd:.3f} s  (avg {1000*d_pd/d_step:.2f} ms/step)")
            print(f"   Pure verify total:  {d_pv:.3f} s  (avg {1000*d_pv/d_step:.2f} ms/step)")
            if d_oth > 0:
                print(f"   Other (prep/etc):   {d_oth:.3f} s  (avg {1000*d_oth/d_step:.2f} ms/step)")
            total = d_pd + d_pv + d_oth
            if total > 0:
                print(f"   Pure draft frac:    {100*d_pd/total:.1f}%")
                print(f"   Pure verify frac:   {100*d_pv/total:.1f}%")
                print(f"   Other frac:         {100*d_oth/total:.1f}%")
    elif t_after:
        n = t_after["step_count"]
        if n > 0:
            print(f"=== Timing Breakdown — {algo_after} (cumulative since server start) ===")
            print(f"   Verify steps:       {n}")
            print(f"   Pure draft total:   {t_after['pure_draft_total_s']:.3f} s  "
                  f"(avg {t_after['pure_draft_avg_ms']:.2f} ms/step)")
            print(f"   Pure verify total:  {t_after['pure_verify_total_s']:.3f} s  "
                  f"(avg {t_after['pure_verify_avg_ms']:.2f} ms/step)")
            if 'other_total_s' in t_after:
                print(f"   Other (prep/etc):   {t_after['other_total_s']:.3f} s  "
                      f"(avg {t_after['other_avg_ms']:.2f} ms/step)")
    else:
        print("NOTE: no dflash_timings returned by /server_info")
        print("      (Server may not have the scheduler.py patch applied)")
    print("=" * 60)


if __name__ == "__main__":
    main()
