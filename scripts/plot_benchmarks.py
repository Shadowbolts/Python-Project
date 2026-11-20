"""
Generate speedup-vs-workers plot for Lab 05.
Saves plot to docs/speedup_plot.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

with open("benchmark_results.json") as f:
    r12 = json.load(f)
with open("benchmark_313t_results.json") as f:
    r13 = json.load(f)

# Extract data
workers_threads = [1] + [row["workers"] for row in r12 if row["executor"] == "threads"]
speedup_threads = [1.0] + [row["speedup"] for row in r12 if row["executor"] == "threads"]

workers_procs = [1] + [row["workers"] for row in r12 if row["executor"] == "processes"]
speedup_procs = [1.0] + [row["speedup"] for row in r12 if row["executor"] == "processes"]

workers_313t = [row["workers"] for row in r13]
speedup_313t = [row["speedup"] for row in r13]

all_workers = [1, 2, 4, 8, 16]
ideal = [float(w) for w in all_workers]

plt.figure(figsize=(9, 6), dpi=150)
plt.plot(all_workers, ideal, "k--", alpha=0.5, label="Ideal linear speedup")
plt.plot(workers_procs, speedup_procs, "o-", color="#2ca02c", linewidth=2.5, markersize=7, label="Processes (Python 3.12, multi-process)")
plt.plot(workers_313t, speedup_313t, "s-", color="#1f77b4", linewidth=2.5, markersize=7, label="Threads (Python 3.13t, free-threaded / no GIL)")
plt.plot(workers_threads, speedup_threads, "x-", color="#d62728", linewidth=2.0, markersize=7, label="Threads (Python 3.12, with GIL)")

plt.title("Indexing Speedup vs Number of Workers (Corpus: 43 books, 65 MB)\nIntel Core i5-13400 (10 cores, 16 threads)", fontsize=13, pad=12)
plt.xlabel("Number of Workers", fontsize=11)
plt.ylabel("Speedup (relative to 1 worker)", fontsize=11)
plt.xticks(all_workers)
plt.xlim(0.8, 16.5)
plt.ylim(0.5, 6.0)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left", fontsize=10)

out_dir = Path("docs")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "speedup_plot.png"
plt.tight_layout()
plt.savefig(out_path)
print(f"Plot successfully saved to {out_path}")
