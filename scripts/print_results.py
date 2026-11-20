import json

with open("benchmark_results.json") as f:
    r12 = json.load(f)
with open("benchmark_313t_results.json") as f:
    r13 = json.load(f)

print("=== 3.12 Standard (with GIL) ===")
for row in r12:
    print(f"{row['executor']:<10} {row['workers']:<3} wall={row['wall_median']:<6.3f}s cpu={row['cpu_median']:<6.3f}s merge={row['merge_median']:<5.3f}s rss={row['rss_median']:<5.1f}MB speedup={row['speedup']:<4.2f}x")

print("\n=== 3.13 Free-Threaded (No GIL) ===")
for row in r13:
    print(f"threads_313t {row['workers']:<3} wall={row['wall_median']:<6.3f}s merge={row['merge_median']:<5.3f}s speedup={row['speedup']:<4.2f}x")
