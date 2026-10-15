#!/usr/bin/env python3
import json, sys, glob, os, statistics as st
R = os.environ.get("RUNS_DIR", "runs")
files = sys.argv[1:] or sorted(glob.glob(R + "/*.json"))
summary = {}
for f in files:
    j = json.load(open(f)); arm = j["arm"]
    rows = []
    for r in j["decode"]:
        ct = (r["usage"] or {}).get("completion_tokens") or 0
        dec = (ct - 1) / (r["t_last"] - r["t_first"]) if r["t_first"] and r["t_last"] > r["t_first"] else None
        sd = r.get("spec_delta", {})
        drafts = sum(v for k, v in sd.items() if k.startswith("vllm:spec_decode_num_drafts{") or k.startswith("vllm:spec_decode_num_drafts_total{") or k.startswith("vllm:spec_decode_num_drafts{"))
        dt = sum(v for k, v in sd.items() if k.startswith("vllm:spec_decode_num_draft_tokens{") or k.startswith("vllm:spec_decode_num_draft_tokens_total"))
        acc = sum(v for k, v in sd.items() if k.startswith("vllm:spec_decode_num_accepted_tokens{") or k.startswith("vllm:spec_decode_num_accepted_tokens_total"))
        pos = {}
        for k, v in sd.items():
            if "per_pos" in k and "position=" in k:
                p = int(k.split('position="')[1].split('"')[0]); pos[p] = pos.get(p, 0) + v
        al = 1 + acc / drafts if drafts else None
        rows.append({"domain": r["domain"], "T": r["temperature"], "ct": ct, "finish": r["finish"],
                     "ttft": r["t_first"] - r["t_start"], "decode_tps": dec, "e2e_tps": ct / (r["t_end"] - r["t_start"]),
                     "accept_len": al, "pos_rate": [round(pos[p] / drafts, 3) for p in sorted(pos)] if drafts else None})
    g = [x for x in rows if x["T"] == 0.0]; s = [x for x in rows if x["T"] == 1.0]
    pf = [{"prompt_tokens": p["prompt_tokens"], "ttft": (p["t_first"] or p["t_end"]) - p["t_start"],
           "prefill_tps": p["prompt_tokens"] / ((p["t_first"] or p["t_end"]) - p["t_start"])} for p in j["prefill"]]
    summary[arm] = {"greedy_decode_tps_mean": st.mean(x["decode_tps"] for x in g) if g else None,
                    "sampled_decode_tps_mean": st.mean(x["decode_tps"] for x in s) if s else None,
                    "rows": rows, "prefill": pf, "done": j.get("done", False)}
    print(f"\n== {arm}  greedy mean decode {summary[arm]['greedy_decode_tps_mean']:.2f} tok/s | sampled mean {summary[arm]['sampled_decode_tps_mean'] or 0:.2f}")
    for x in rows:
        print(f"  {x['domain']:8s} T={x['T']} ct={x['ct']} {x['finish']} decode={x['decode_tps']:.2f} e2e={x['e2e_tps']:.2f} ttft={x['ttft']:.2f} AL={x['accept_len'] and round(x['accept_len'],2)} pos={x['pos_rate']}")
    for p in pf:
        print(f"  prefill {p['prompt_tokens']:>7} tok ttft={p['ttft']:.2f}s {p['prefill_tps']:.0f} tok/s")
json.dump(summary, open(R + "/../summary.json", "w"), indent=1)
