#!/usr/bin/env python3
"""Kill the serving container if host MemAvailable < 5 GiB or host swap grows > 8 GiB from baseline.
While the container is running but not healthy (weight load), advise clean model-file pages out
of page cache every 2s so streaming 70 GB of shards does not push anon pages to swap."""
import subprocess, time, json, os, glob, urllib.request, sys
CT = sys.argv[1] if len(sys.argv) > 1 else "nemotron35-super"
LOG = os.environ.get("WATCHDOG_LOG", "watchdog.jsonl")
FILES = glob.glob(os.path.expanduser(os.environ.get("MODELS", "~/models")) + "/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4*/*.safetensors")
def mem():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":"); d[k] = int(v.split()[0]) * 1024
    return d["MemAvailable"], d["SwapTotal"] - d["SwapFree"]
def healthy():
    try: return urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).status == 200
    except Exception: return False
def advise():
    for p in FILES:
        try:
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except Exception: pass
_, base = mem()
low = 1 << 62; last_adv = 0
while True:
    avail, swap = mem(); low = min(low, avail)
    running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CT], capture_output=True, text=True).stdout.strip() == "true"
    h = running and healthy()
    if running and not h and time.time() - last_adv > 2: advise(); last_adv = time.time()
    rec = {"t": time.time(), "avail_gib": round(avail / 2**30, 2), "swap_growth_gib": round((swap - base) / 2**30, 2), "min_avail_gib": round(low / 2**30, 2), "running": running, "healthy": h}
    reason = None
    if avail < 5 * 2**30: reason = "reserve"
    elif swap - base > 8 * 2**30: reason = "swap"
    if reason and running:
        subprocess.run(["docker", "kill", CT]); rec["killed"] = reason
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
    time.sleep(1 if running else 5)
