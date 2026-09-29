"""Run the whole reproduction unattended on several GPUs.

    python scripts/run_suite.py --gpus 0 1 2 3 --jobs_per_gpu 2 --wandb afea-net

Jobs are queued in priority order so that the most important results finish first:
  1. main      Table 4/6 ablations (12 configs), first seed
  2. seeds     the same 12 configs for the remaining seeds
  3. variants  full model with D=1024 BiLSTM and without validation-based selection (all seeds)
  4. sweeps    margin / alpha / beta / gamma sweeps of Fig. 4-5 (first seed)
A job is one (dataset, config, seed) = 5-fold CV in one process. Free GPU slots pull the next
job from a shared queue. Finished jobs (summary.json present) are skipped and finished folds
inside a job are resumed, so the command can simply be re-run after any interruption.
Failed jobs are retried once at the end.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

import _path  # noqa: F401

from afea.suite import ablation_configs, diagnosis_configs, sweep_configs, variant_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_jobs(args):
    datasets = [d for d in args.datasets
                if os.path.exists(os.path.join(ROOT, "manifests", f"{d}.csv"))]
    for d in set(args.datasets) - set(datasets):
        print(f"WARNING: manifests/{d}.csv not found, skipping dataset {d}")
    first, rest = args.seeds[0], args.seeds[1:]
    jobs = []
    for tier in args.tiers:  # tiers run in the order given on the command line
        if tier == "main":
            jobs += [(d, n, first, a) for d in datasets for n, a in ablation_configs(d)]
        elif tier == "seeds":
            jobs += [(d, n, s, a) for s in rest for d in datasets for n, a in ablation_configs(d)]
        elif tier == "variants":
            jobs += [(d, n, s, a) for s in args.seeds for d in datasets for n, a in variant_configs(d)]
        elif tier == "sweeps":
            jobs += [(d, n, first, a) for d in datasets for n, a in sweep_configs(d)]
        elif tier == "diag":
            for d in datasets:
                if d != "iemocap":
                    continue
                has_all = os.path.isdir(os.path.join(ROOT, "features", d, "wavlm_all"))
                for n, a, _ in diagnosis_configs(d):
                    if "all" in a and not has_all:
                        print(f"WARNING: features/{d}/wavlm_all missing, skipping {n}")
                        continue
                    jobs += [(d, n, s, a) for s in args.diag_seeds]
    return jobs


def out_dir(d, name, seed):
    return os.path.join(ROOT, "runs", d, name, f"seed{seed}")


def command(job, args):
    d, name, seed, cfg_args = job
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--dataset", d,
           "--manifest", os.path.join(ROOT, "manifests", f"{d}.csv"),
           "--feat_root", os.path.join(ROOT, "features", d),
           "--out", out_dir(d, name, seed), "--seeds", str(seed), "--cache"]
    if args.wandb:
        cmd += ["--wandb", args.wandb, "--wandb_group", f"{d}_{name}", "--wandb_name", f"{d}_{name}_s{seed}"]
    # global extras first, config-specific args last so they win (argparse keeps the last value)
    return cmd + args.extra + cfg_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--jobs_per_gpu", type=int, default=2)
    ap.add_argument("--datasets", nargs="+", default=["iemocap", "ravdess"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--tiers", nargs="+", default=["main", "seeds", "variants", "sweeps"],
                    choices=["main", "seeds", "variants", "sweeps", "diag"])
    ap.add_argument("--diag_seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--stop_after_hours", type=float, default=None,
                    help="do not START new jobs after this many hours (running jobs finish)")
    ap.add_argument("--threads_per_job", type=int, default=4, help="OMP/MKL threads per process")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --, passed to every train.py call")
    args = ap.parse_args()
    args.extra = [a for a in args.extra if a != "--"]

    jobs = build_jobs(args)
    todo = [j for j in jobs if not os.path.exists(os.path.join(out_dir(j[0], j[1], j[2]), "summary.json"))]
    print(f"{len(jobs)} jobs in suite, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"on {len(args.gpus)} GPU(s) x {args.jobs_per_gpu} slots")
    if args.dry_run:
        for j in todo:
            print(f"  {j[0]:8s} {j[1]:22s} seed {j[2]:3d}  {' '.join(j[3])}")
        return

    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    status_path = os.path.join(ROOT, "runs", "suite_status.log")
    lock = threading.Lock()
    t_start = time.time()
    done_count = [0]

    def status(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with lock:
            print(line, flush=True)
            with open(status_path, "a") as f:
                f.write(line + "\n")

    def run_queue(job_list):
        q = queue.Queue()
        for j in job_list:
            q.put(j)
        failed = []

        def worker(gpu):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       OMP_NUM_THREADS=str(args.threads_per_job), MKL_NUM_THREADS=str(args.threads_per_job))
            while True:
                if args.stop_after_hours and time.time() - t_start > args.stop_after_hours * 3600:
                    return
                try:
                    job = q.get_nowait()
                except queue.Empty:
                    return
                d, name, seed, _ = job
                od = out_dir(d, name, seed)
                os.makedirs(od, exist_ok=True)
                status(f"START gpu{gpu} {d}/{name}/seed{seed}")
                t0 = time.time()
                with open(os.path.join(od, "stdout.log"), "a") as log:
                    rc = subprocess.call(command(job, args), stdout=log, stderr=subprocess.STDOUT, env=env)
                ok = rc == 0 and os.path.exists(os.path.join(od, "summary.json"))
                with lock:
                    done_count[0] += 1
                status(f"{'DONE ' if ok else 'FAIL '} gpu{gpu} {d}/{name}/seed{seed} "
                       f"({(time.time() - t0) / 60:.1f} min, rc={rc}) [{done_count[0]}/{len(todo)}]")
                if not ok:
                    with lock:
                        failed.append(job)

        threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.jobs_per_gpu)]
        for i, t in enumerate(threads):
            t.start()
            time.sleep(2 if i < len(threads) - 1 else 0)  # stagger start-up I/O
        for t in threads:
            t.join()
        return failed

    failed = run_queue(todo)
    if failed:
        status(f"retrying {len(failed)} failed job(s) once")
        failed = run_queue(failed)
    status(f"suite finished in {(time.time() - t_start) / 3600:.2f} h; "
           f"{len(failed)} job(s) still failing: {[f'{j[0]}/{j[1]}/seed{j[2]}' for j in failed]}")
    subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "aggregate.py")])
    subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "diagnose.py")])


if __name__ == "__main__":
    main()
