from pathlib import Path
import pandas as pd

benchmark_file = Path(
    "data/benchmarks/NFS_benchmarks.csv"
)

if not benchmark_file.exists():
    raise FileNotFoundError(
        "NFS_benchmarks.csv not found"
    )

df = pd.read_csv(benchmark_file)

print("=" * 60)
print("PHASE 1.6 NFS BENCHMARK VERIFICATION")
print("=" * 60)

print("\nRows :", len(df))
print("Columns :", len(df.columns))

print("\nColumn Names:")
print(list(df.columns))

print("\nFirst Five Rows:")
print(df.head())

print("\nBenchmark File Verified")