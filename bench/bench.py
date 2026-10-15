#!/usr/bin/env python3
"""Nemotron 3.5 Super single-GB10 bench. stdlib only.
decode: 4 domains x fixed 512 forced output tokens, single stream, greedy + one sampled.
prefill: cold (unique nonce at prompt start) 2K/8K/32K real-text prompts, max_tokens=1.
Writes one JSON per arm with raw rows; derived rates computed after persistence.
"""
import argparse, json, os, re, time, urllib.request, uuid, glob, random

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MODEL = "nemotron-3.5-super-vl"

DECODE = {
    "code": "Write a complete, well-commented Python module implementing an LRU cache with TTL expiry, thread safety, and a decorator interface. Include unit tests using unittest.",
    "prose": "Write a long, vivid short story about a lighthouse keeper on a remote island who discovers that the light is signalling to something beneath the sea.",
    "math": "Solve step by step: a tank is filled by pipe A in 6 hours and pipe B in 9 hours, and drained by pipe C in 12 hours. Starting empty with all three open, after 2 hours pipe C is closed. How long in total until the tank is full? Then generalize the method.",
    "summary": None,  # filled from a source document
}

def post(path, body, stream=False, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def metrics():
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        m = re.match(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m:
            key = m.group(1) + (m.group(2) or "")
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out

def stream_chat(messages, max_tokens, temperature=0.0, top_p=1.0, ignore_eos=True, thinking=True):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "stream": True, "stream_options": {"include_usage": True},
            "ignore_eos": ignore_eos, "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    t0 = time.perf_counter(); first = None; last = None; events = 0; usage = None
    text = []; reasoning = []; finish = None
    with post("/v1/chat/completions", body, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            for c in j.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if delta.get("content"): text.append(delta["content"])
                if delta.get("reasoning_content") or delta.get("reasoning"): reasoning.append(delta.get("reasoning_content") or delta.get("reasoning"))
                if piece:
                    now = time.perf_counter(); events += 1
                    if first is None: first = now
                    last = now
                if c.get("finish_reason"): finish = c["finish_reason"]
    t1 = time.perf_counter()
    return {"t_start": t0, "t_first": first, "t_last": last, "t_end": t1, "events": events, "usage": usage,
            "finish": finish, "content": "".join(text), "reasoning": "".join(reasoning)}

def source_text():
    # Prompt corpus: any large text tree. Measured with a llama.cpp b9222 checkout (src/*.cpp + docs/**/*.md).
    root = os.environ.get("CORPUS_DIR", os.path.expanduser("~/llama.cpp"))
    files = sorted(glob.glob(root + "/src/*.cpp")) + sorted(glob.glob(root + "/docs/**/*.md", recursive=True))
    buf = []
    for f in files:
        try: buf.append(open(f, errors="ignore").read())
        except Exception: pass
    return "\n\n".join(buf)

def tokenize(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.loads(r.read())["count"]

def build_prompt(src, target_tokens, nonce):
    # binary search char length to hit target tokens (raw text, excluding template)
    lo, hi = 1000, min(len(src), target_tokens * 8)
    best = None
    for _ in range(14):
        mid = (lo + hi) // 2
        n = tokenize(nonce + src[:mid])
        if n <= target_tokens: best = mid; lo = mid + 1
        else: hi = mid - 1
    return nonce + src[:best]

def run(arm, out, prefill_sizes, reps):
    res = {"arm": arm, "time": time.time(), "decode": [], "prefill": []}
    src = source_text()
    DECODE["summary"] = "Summarize the following source code in a detailed technical report, section by section:\n\n" + src[200000:212000]
    # warmup (excluded)
    stream_chat([{"role": "user", "content": "Say hello."}], 32, ignore_eos=False)
    # decode rows
    for rep in range(reps):
        for dom, prompt in DECODE.items():
            for temp, top_p in ((0.0, 1.0),) + (((1.0, 0.95),) if rep == 0 else ()):
                m0 = metrics()
                r = stream_chat([{"role": "user", "content": prompt}], 512, temperature=temp, top_p=top_p)
                m1 = metrics()
                r.update(domain=dom, temperature=temp, rep=rep, spec_delta={k: m1.get(k, 0) - m0.get(k, 0) for k in m1})
                res["decode"].append(r)
                json.dump(res, open(out, "w"), indent=1)
                ct = (r["usage"] or {}).get("completion_tokens")
                dec = (r["events"] - 1) / (r["t_last"] - r["t_first"]) if r["events"] > 1 else None
                e2e = ct / (r["t_end"] - r["t_start"]) if ct else None
                print(f"[{arm}] decode {dom} T={temp} rep{rep}: tokens={ct} ttft={r['t_first']-r['t_start']:.3f}s e2e={e2e and round(e2e,2)} tok/s finish={r['finish']}", flush=True)
    # prefill rows (cold: unique nonce first)
    for n in prefill_sizes:
        nonce = f"[session {uuid.uuid4().hex}]\n"
        p = build_prompt(src, n, nonce)
        r = stream_chat([{"role": "user", "content": p + "\n\nIn one word, what language is this?"}], 1, ignore_eos=False, thinking=False)
        pt = (r["usage"] or {}).get("prompt_tokens")
        r.update(target=n, prompt_tokens=pt); r.pop("content"); r.pop("reasoning")
        res["prefill"].append(r)
        json.dump(res, open(out, "w"), indent=1)
        ttft = r["t_first"] - r["t_start"] if r["t_first"] else r["t_end"] - r["t_start"]
        print(f"[{arm}] prefill {pt} tok: ttft={ttft:.3f}s -> {pt/ttft:.0f} tok/s", flush=True)
    res["done"] = True
    json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prefill", default="2048,8192,32768"); ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    run(a.arm, a.out, [int(x) for x in a.prefill.split(",") if x], a.reps)
