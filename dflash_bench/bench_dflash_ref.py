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


def compute_flops_per_token(model_path, avg_seq_len=512):
    """Exact FLOPs per decoded token (GQA + SwiGLU + attention + lm_head)."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    h = c.hidden_size
    L = c.num_hidden_layers
    n_heads = c.num_attention_heads
    n_kv = getattr(c, "num_key_value_heads", n_heads)
    d_ff = c.intermediate_size
    V = c.vocab_size
    head_dim = h // n_heads
    h_kv = head_dim * n_kv

    flops_q   = 2 * h * h
    flops_k   = 2 * h * h_kv
    flops_v   = 2 * h * h_kv
    flops_o   = 2 * h * h
    flops_ffn = 2 * h * d_ff * 3       # SwiGLU: gate, up, down
    flops_attn = 4 * h * avg_seq_len   # QK^T + softmax-V
    per_layer = flops_q + flops_k + flops_v + flops_o + flops_ffn + flops_attn
    flops_layers = L * per_layer
    flops_lm_head = 2 * h * V
    total = flops_layers + flops_lm_head

    breakdown = {
        "hidden_size": h,
        "layers": L,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "intermediate_size": d_ff,
        "vocab_size": V,
        "avg_seq_len": avg_seq_len,
        "per_layer_FLOPs": per_layer,
        "all_layers_FLOPs": flops_layers,
        "lm_head_FLOPs": flops_lm_head,
        "total_FLOPs_per_token": total,
    }
    return total, breakdown


def compute_mfu(aggregate_throughput_tok_s, flops_per_token,
                peak_tflops_per_chip, tp):
    """MFU = F_t * aggregate_throughput / peak.

    Works for any concurrency. At concurrency=1, this equals F_t / ITL / peak.
    At higher concurrency, aggregate throughput captures all concurrent streams,
    while per-stream ITL would undercount by the concurrency factor.

    Args:
        aggregate_throughput_tok_s: total output tokens/sec across all streams
        flops_per_token: target FLOPs per accepted output token
        peak_tflops_per_chip: hardware peak BF16 TFLOPS per chip
        tp: tensor parallel size

    Returns:
        MFU as a fraction (0-1).
    """
    peak_flops = peak_tflops_per_chip * 1e12 * tp
    flops_per_sec = flops_per_token * aggregate_throughput_tok_s
    print(f"####################  flops_per_sec: {flops_per_token} ####################")
    return flops_per_sec / peak_flops


def send_one(base_url, prompt, max_new_tokens, temperature, top_p, top_k, timeout):
    """Streaming request to /generate. Captures per-token timing the same way
    sglang.bench_serving does: each SSE chunk records a timestamp, and per-token
    ITL = chunk_gap / num_new_tokens (since multiple tokens can land per chunk
    when spec decoding accepts a block).

    Returns dict with keys:
        text, meta_info, e2e_latency, ttft, itls (list, ms), tpot_ms, output_len
    """
    import json as _json
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
        },
        "stream": True,
    }

    st = time.perf_counter()
    ttft = 0.0
    most_recent_ts = st
    last_output_len = 0
    itls = []
    final_data = None

    with requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=timeout,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].lstrip()
            if line == "[DONE]":
                continue
            try:
                data = _json.loads(line)
            except Exception:
                continue
            if "text" not in data or "meta_info" not in data:
                continue
            output_len = int(data["meta_info"].get("completion_tokens", 0))
            ts = time.perf_counter()
            if ttft == 0.0 and output_len > 0:
                ttft = ts - st
            else:
                num_new = output_len - last_output_len
                if num_new <= 0:
                    continue
                chunk_gap = ts - most_recent_ts
                per_token_itl = chunk_gap / num_new
                itls.extend([per_token_itl] * num_new)
            most_recent_ts = ts
            last_output_len = output_len
            final_data = data

    e2e = time.perf_counter() - st
    output_len = (final_data or {}).get("meta_info", {}).get("completion_tokens", 0)
    tpot = ((e2e - ttft) / max(1, output_len - 1)) if output_len > 1 else 0.0

    if final_data is None:
        # Server returned nothing useful — surface an empty record
        return {
            "text": "",
            "meta_info": {},
            "e2e_latency": e2e,
            "ttft": ttft,
            "itls": [],
            "tpot_ms": 0.0,
            "output_len": 0,
        }

    out = {
        "text": final_data.get("text", ""),
        "meta_info": final_data.get("meta_info", {}),
        "e2e_latency": e2e,
        "ttft": ttft,
        "itls": [x * 1000.0 for x in itls],  # ms
        "tpot_ms": tpot * 1000.0,
        "output_len": output_len,
    }
    return out


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
    ap.add_argument("--peak-tflops-per-chip", type=float, default=320.0,
                    help="Peak BF16 TFLOPS per chip. Ascend 910B = 320 (BF16 spec).")
    ap.add_argument("--tp", type=int, default=8,
                    help="Tensor parallel size for aggregate peak FLOPS calc.")
    ap.add_argument("--avg-seq-len", type=int, default=512,
                    help="Average seq len for attention FLOPs. ~1024 reasonable for GSM8K outputs.")
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
    all_itls = []     # per-token ITL in ms, aggregated across requests
    all_ttfts = []    # per-request TTFT in ms
    all_tpots = []    # per-request TPOT in ms

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
            # Streaming-derived latency metrics
            req_itls = out.get("itls") or []
            all_itls.extend(req_itls)
            ttft_ms = (out.get("ttft", 0.0) or 0.0) * 1000.0
            if ttft_ms > 0:
                all_ttfts.append(ttft_ms)
            tpot_ms = out.get("tpot_ms", 0.0)
            if tpot_ms > 0:
                all_tpots.append(tpot_ms)
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
    if verify_ct_sum:
        print(f"   Verify ct sum:      {verify_ct_sum}")
    if e2e_latencies:
        print(f"   E2E latency (mean): {statistics.mean(e2e_latencies):.2f} s")
        print(f"   E2E latency (min):  {min(e2e_latencies):.2f} s")
        print(f"   E2E latency (max):  {max(e2e_latencies):.2f} s")
    print()

    # Streaming-derived per-token metrics (same algorithm as sglang.bench_serving)
    if all_itls:
        import math
        s_itls = sorted(all_itls)
        def _pct(arr, q):
            if not arr:
                return 0.0
            idx = min(len(arr) - 1, max(0, int(math.ceil(q * len(arr))) - 1))
            return arr[idx]
        print(f"--- Per-token latency (streaming, {len(all_itls)} samples) ---")
        print(f"   Mean TTFT (ms):     {statistics.mean(all_ttfts):.2f}" if all_ttfts else "")
        print(f"   Mean ITL (ms):      {statistics.mean(all_itls):.2f}")
        print(f"   Median ITL (ms):    {statistics.median(all_itls):.2f}")
        print(f"   P95 ITL (ms):       {_pct(s_itls, 0.95):.2f}")
        print(f"   P99 ITL (ms):       {_pct(s_itls, 0.99):.2f}")
        print(f"   Max ITL (ms):       {max(all_itls):.2f}")
        if all_tpots:
            print(f"   Mean TPOT (ms):     {statistics.mean(all_tpots):.2f}")
        print()

        # === MFU computation using exact target FLOPs ===
        try:
            flops_per_token, bd = compute_flops_per_token(
                args.model, avg_seq_len=args.avg_seq_len,
            )
            mean_itl_s = statistics.mean(all_itls) / 1000.0
            median_itl_s = statistics.median(all_itls) / 1000.0
            # tau = accepted tokens per spec round (incl. bonus).
            if accept_lengths:
                tau = statistics.mean(accept_lengths) + 1.0  # +1 bonus
            else:
                tau = 1.0
            # round_time = ITL@gamma = wall-clock per spec round per stream.
            # Note: at high concurrency, per-stream ITL is inflated by concurrency.
            round_time_mean_s   = mean_itl_s   * tau
            round_time_median_s = median_itl_s * tau
            # Aggregate throughput = total tokens / wall clock — works at ANY concurrency.
            aggregate_throughput = total_tokens / elapsed
            # MFU using aggregate throughput (correct at any concurrency)
            mfu_aggregate = compute_mfu(aggregate_throughput, flops_per_token,
                                        args.peak_tflops_per_chip, args.tp)
            # MFU using per-stream ITL (only correct at concurrency=1)
            # Kept for backward comparison; flag as wrong at high concurrency.
            peak_flops = args.peak_tflops_per_chip * 1e12 * args.tp
            mfu_mean = (flops_per_token / mean_itl_s) / peak_flops
            mfu_median = (flops_per_token / median_itl_s) / peak_flops
            peak_total = args.peak_tflops_per_chip * args.tp
            bd_h     = bd["hidden_size"]
            bd_L     = bd["layers"]
            bd_V     = bd["vocab_size"]
            bd_kv    = bd["n_kv_heads"]
            bd_ff    = bd["intermediate_size"]
            bd_seq   = bd["avg_seq_len"]
            bd_lm    = bd["lm_head_FLOPs"]
            bd_total = bd["total_FLOPs_per_token"]
            lm_frac  = 100.0 * bd_lm / bd_total
            print(f"--- MFU ---")
            print(f"   Target model:       {args.model}")
            print(f"   F_t (FLOPs/token):  {flops_per_token/1e9:.2f} GFLOPs/token")
            print(f"   tau (accept+bonus): {tau:.3f}")
            print(f"   Aggregate throughput: {aggregate_throughput:.2f} tok/s")
            print(f"   round_time mean (per-stream): {round_time_mean_s*1000:.2f} ms")
            print(f"   Breakdown:")
            print(f"     hidden:           {bd_h}")
            print(f"     layers:           {bd_L}")
            print(f"     vocab:            {bd_V}")
            print(f"     n_kv_heads:       {bd_kv}  (GQA)")
            print(f"     intermediate:     {bd_ff}")
            print(f"     attention @ seq:  {bd_seq}")
            print(f"     LM head FLOPs:    {bd_lm/1e9:.2f} G ({lm_frac:.1f}%)")
            print(f"   Peak FLOPS total:   {peak_total:.0f} TFLOPS (TP={args.tp} x {args.peak_tflops_per_chip})")
            print(f"   ** MFU (aggregate throughput): {100*mfu_aggregate:.4f}% **   <- USE THIS")
            print(f"   ** MFU (from per-stream ITL):  {100*mfu_mean:.4f}% **        <- only valid at conc=1")
            print(f"   ** MFU (from median ITL):      {100*mfu_median:.4f}% **       <- only valid at conc=1")
        except Exception as e:
            print(f"   MFU computation failed: {e}")
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
