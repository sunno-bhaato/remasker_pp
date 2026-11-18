"""Run a grid of benchmark jobs in parallel (one single-threaded process per job).
Each job writes results/raw/<tag>/<job>.json as soon as it finishes; finished jobs are skipped,
so the grid can be interrupted and resumed.

usage: python run_grid.py <tag> --datasets a,b --mechs MCAR,MAR --seeds 0,1 --rates 0.3 \
                          --methods m1,m2 [--workers 10]
"""
import argparse, json, os, sys, time, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
SIZE = {"yacht": 1, "raisin": 3, "compression": 4, "airfoil": 5, "wine": 6}
COST = {"hyperimpute": 8, "missforest": 5, "miwae": 3, "gain": 2, "remasker": 5, "pp": 4, "oracle": 0}


def work(job, outdir):
    d, mech, rate, seed, method = job
    path = os.path.join(outdir, f"{d}_{mech}_{rate}_{seed}_{method}.json")
    if os.path.exists(path):
        return path, "skip"
    # separate process per job: torch+xgboost libomp can abort at interpreter exit,
    # so the job writes its result file itself before exiting.
    for attempt in range(3):  # retry: occasional libomp init failures under heavy parallelism
        p = subprocess.run([sys.executable, os.path.join(ROOT, "bench.py"), d, mech, str(rate), str(seed), method, path],
                           capture_output=True, text=True, cwd=ROOT)
        if os.path.exists(path):
            return path, "ok" if attempt == 0 else f"ok (retry {attempt})"
    return path, "FAIL " + p.stderr[-800:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--datasets", default="yacht,airfoil,compression,raisin,wine")
    ap.add_argument("--mechs", default="MCAR,MAR,MNAR")
    ap.add_argument("--rates", default="0.3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "results", "raw", a.tag)
    os.makedirs(outdir, exist_ok=True)
    jobs = [(d, m, float(r), int(s), meth)
            for d in a.datasets.split(",") for m in a.mechs.split(",") for r in a.rates.split(",")
            for s in a.seeds.split(",") for meth in a.methods.split(",")]
    if "oracle" in a.methods:
        jobs = [j for j in jobs if j[4] != "oracle"] + [(d, "none", 0.0, int(s), "oracle")
                                                       for d in a.datasets.split(",") for s in a.seeds.split(",")]
    jobs.sort(key=lambda j: -SIZE.get(j[0], 3) * COST.get(j[4].split("_")[0] if not j[4].startswith("pp") else "pp", 3))
    t0 = time.time()
    print(f"{len(jobs)} jobs -> {outdir}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, j, outdir) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            p, status = f.result()
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:6.0f}s] {os.path.basename(p)} {status}", flush=True)
