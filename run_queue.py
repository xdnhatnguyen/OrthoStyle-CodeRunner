#!/usr/bin/env python3
"""
OrthoStyle Master Task Queue Runner (Multi-GPU Dynamic Worker Pool)
Automatically distributes experiments across available GPUs (e.g. 2 GPUs).
Whenever a GPU finishes a task, it immediately picks the next task from the queue,
keeping all GPUs 100% utilized until the entire suite is complete.

Usage:
    python run_queue.py --suite all --gpus 0 1
    python run_queue.py --suite ablations --gpus 0 1
    python run_queue.py --suite sweeps --gpus 0 1
    python run_queue.py --suite table2 --gpus 0 1
    python run_queue.py --suite main --gpus 0 1
"""

import argparse
import datetime
import os
import queue
import subprocess
import sys
import threading
import time

PYTHON_BIN = sys.executable

# -----------------------------------------------------------------------------
# Experiment Task Definitions
# -----------------------------------------------------------------------------

# 1. Tau Pushforward Sweep [1, 2, 3, 4] on first 7 contents x 7 styles (49 pairs)
TASKS_SWEEPS = [
    {
        "name": "sweep_tau_1",
        "desc": "Tau Sweep: tau=1 (p_switch=0.05, 49 pairs 7x7)",
        "args": ["--tau_pushforward", "1", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "sweep_tau_1"],
    },
    {
        "name": "sweep_tau_2",
        "desc": "Tau Sweep: tau=2 (p_switch=0.10, 49 pairs 7x7 - default)",
        "args": ["--tau_pushforward", "2", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "sweep_tau_2"],
    },
    {
        "name": "sweep_tau_3",
        "desc": "Tau Sweep: tau=3 (p_switch=0.15, 49 pairs 7x7)",
        "args": ["--tau_pushforward", "3", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "sweep_tau_3"],
    },
    {
        "name": "sweep_tau_4",
        "desc": "Tau Sweep: tau=4 (p_switch=0.20, 49 pairs 7x7)",
        "args": ["--tau_pushforward", "4", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "sweep_tau_4"],
    },
]

# 2. Table 2: Component Ablation Study (Configurations B to F on 49 pairs 7x7)
TASKS_TABLE2 = [
    {
        "name": "ablation_B_pure_mean",
        "desc": "Table 2 (B): Pure Mean Token (alpha_s=0.0, 49 pairs 7x7)",
        "args": ["--alpha_style", "0.0", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "ablation_B_pure_mean"],
    },
    {
        "name": "ablation_C_raw_style",
        "desc": "Table 2 (C): Raw Style Token (alpha_s=1.0, 49 pairs 7x7)",
        "args": ["--alpha_style", "1.0", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "ablation_C_raw_style"],
    },
    {
        "name": "ablation_D_no_ortho",
        "desc": "Table 2 (D): No Score-Orthogonal Guidance (49 pairs 7x7)",
        "args": ["--no_ortho", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "ablation_D_no_ortho"],
    },
    {
        "name": "ablation_E_no_pushforward",
        "desc": "Table 2 (E): No AdaIN Pushforward (tau=0, p_switch=0, 49 pairs 7x7)",
        "args": ["--no_pushforward", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "ablation_E_no_pushforward"],
    },
    {
        "name": "ablation_F_no_semantic_gating",
        "desc": "Table 2 (F): No Semantic Gated Canny (49 pairs 7x7)",
        "args": ["--no_semantic_gating", "--subset_7x7", "--prompt_levels", "null", "--ablation_tag", "ablation_F_no_semantic_gating"],
    },
]

# 3. Main Benchmark Comparison: Level 1 Null Prompt (Full 225 pairs split across 2 workers)
TASKS_MAIN_BENCHMARK = [
    {
        "name": "benchmark_null_part1",
        "desc": "Main Benchmark Level 1 Null Prompt (Pairs 1..112)",
        "args": ["--start_idx", "1", "--end_idx", "112", "--prompt_levels", "null"],
    },
    {
        "name": "benchmark_null_part2",
        "desc": "Main Benchmark Level 1 Null Prompt (Pairs 113..225)",
        "args": ["--start_idx", "113", "--end_idx", "225", "--prompt_levels", "null"],
    },
]

# 4. Prompt Levels 2 & 3 (Object & Style Description across 225 pairs)
TASKS_OTHER_LEVELS = [
    {
        "name": "benchmark_object_part1",
        "desc": "Benchmark Level 2 Object Prompt (Pairs 1..112)",
        "args": ["--start_idx", "1", "--end_idx", "112", "--prompt_levels", "object"],
    },
    {
        "name": "benchmark_object_part2",
        "desc": "Benchmark Level 2 Object Prompt (Pairs 113..225)",
        "args": ["--start_idx", "113", "--end_idx", "225", "--prompt_levels", "object"],
    },
    {
        "name": "benchmark_style_desc_part1",
        "desc": "Benchmark Level 3 Style Description (Pairs 1..112)",
        "args": ["--start_idx", "1", "--end_idx", "112", "--prompt_levels", "style_desc"],
    },
    {
        "name": "benchmark_style_desc_part2",
        "desc": "Benchmark Level 3 Style Description (Pairs 113..225)",
        "args": ["--start_idx", "113", "--end_idx", "225", "--prompt_levels", "style_desc"],
    },
]

SUITE_MAP = {
    "sweeps": TASKS_SWEEPS,
    "table2": TASKS_TABLE2,
    "ablations": TASKS_SWEEPS + TASKS_TABLE2,
    "main": TASKS_MAIN_BENCHMARK,
    "levels": TASKS_OTHER_LEVELS,
    "all": TASKS_SWEEPS + TASKS_TABLE2 + TASKS_MAIN_BENCHMARK + TASKS_OTHER_LEVELS,
}


# -----------------------------------------------------------------------------
# Worker Thread & Dynamic Dispatch
# -----------------------------------------------------------------------------

def worker_loop(gpu_id: str, task_queue: queue.Queue, results: list, lock: threading.Lock, total_tasks: int):
    device_str = f"cuda:{gpu_id}" if gpu_id.isdigit() else gpu_id

    while True:
        try:
            task_idx, task = task_queue.get_nowait()
        except queue.Empty:
            break

        task_name = task["name"]
        task_desc = task["desc"]
        log_file = os.path.join("logs", f"{task_name}.log")

        with lock:
            print(f"\n{'='*70}")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [TASK {task_idx}/{total_tasks}] ASSIGNED TO {device_str.upper()}")
            print(f"  Name:        {task_name}")
            print(f"  Description: {task_desc}")
            print(f"  Log File:    {log_file}")
            print(f"{'='*70}\n")

        cmd = [PYTHON_BIN, "batch_runner.py", "--device", device_str] + task["args"]

        start_time = time.time()
        success = False
        error_msg = ""

        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy()
                )
                ret_code = proc.wait()
                if ret_code == 0:
                    success = True
                else:
                    error_msg = f"Exit code {ret_code}"
        except Exception as e:
            error_msg = str(e)

        elapsed = time.time() - start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

        with lock:
            status_tag = "[✔ SUCCESS]" if success else f"[✘ FAILED: {error_msg}]"
            print(f"\n{status_tag} Task {task_name} on {device_str.upper()} finished in {elapsed_str}.")
            results.append({
                "task_idx": task_idx,
                "name": task_name,
                "gpu": device_str,
                "elapsed": elapsed_str,
                "success": success,
                "error": error_msg,
                "log": log_file
            })

        task_queue.task_done()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OrthoStyle Dynamic Multi-GPU Task Queue Runner")
    parser.add_argument("--suite", type=str, default="all", choices=list(SUITE_MAP.keys()),
                        help="Suite to run: 'all', 'ablations', 'sweeps', 'table2', 'main', 'levels'")
    parser.add_argument("--gpus", nargs="+", default=["0", "1"],
                        help="GPU IDs to utilize concurrently (e.g. --gpus 0 1)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("output/benchmark", exist_ok=True)

    task_list = SUITE_MAP[args.suite]
    total_tasks = len(task_list)

    print("=" * 70)
    print("  OrthoStyle Multi-GPU Dynamic Queue Dispatcher")
    print(f"  Suite Selected:   {args.suite.upper()} ({total_tasks} total tasks)")
    print(f"  GPUs Allocated:   {args.gpus}")
    print(f"  Python Bin:       {PYTHON_BIN}")
    print(f"  Start Time:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    task_queue = queue.Queue()
    for idx, task in enumerate(task_list, start=1):
        task_queue.put((idx, task))

    results = []
    lock = threading.Lock()
    threads = []

    overall_start = time.time()

    for gpu_id in args.gpus:
        t = threading.Thread(
            target=worker_loop,
            args=(gpu_id, task_queue, results, lock, total_tasks),
            daemon=False
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    overall_elapsed = time.time() - overall_start
    overall_elapsed_str = str(datetime.timedelta(seconds=int(overall_elapsed)))

    # Print Summary Table
    print("\n" + "=" * 75)
    print("                       ALL EXPERIMENTS COMPLETED")
    print(f"  Total Duration: {overall_elapsed_str}")
    print("=" * 75)
    print(f"{'#':<3} | {'Task Name':<30} | {'GPU':<8} | {'Duration':<10} | {'Status'}")
    print("-" * 75)
    results.sort(key=lambda r: r["task_idx"])
    for r in results:
        status = "PASSED" if r["success"] else f"FAILED ({r['error']})"
        print(f"{r['task_idx']:<3} | {r['name']:<30} | {r['gpu']:<8} | {r['elapsed']:<10} | {status}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
