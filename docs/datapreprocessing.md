# Data Preprocessing — Input files and notes

This short doc lists the key files used by the preprocessing pipeline and what they contain. It is meant to live in `docs/` as a quick reference for running and understanding the pipeline.

## Key input files (locations and purpose)

- `archive/UCI_Credit_Card.csv` — Raw UCI credit card dataset (source CSV). Primary raw input for preprocessing.
- `configs/credit_preprocessing_config.json` — Pipeline configuration (paths, output formats, parallel/MPI flags, test split, etc.). Edit this to change input/output behavior.
- `src/credit_preprocessing.py` — Preprocessing code (data cleaning, imputation, feature engineering). The script reads the raw dataset (or configured input) and writes outputs.
- `src/run_preprocessing_pipeline.py` — CLI/runner that loads the config and executes the preprocessing steps; accepts an `--mpi` flag if MPI is available.
- `scripts/run_preprocessing.sh` — Convenience shell wrapper (if present) to run the pipeline with default settings.

## Typical outputs (where to find processed artifacts)

- `processed_data/cleaned_data.csv` — Cleaned data saved as CSV (human-readable, easy for quick inspection).
- `processed_data/engineered_features.csv` — Feature-engineered dataset saved as CSV.
- `train/train.parquet`, `test/test.parquet` — Train/test splits saved in Parquet (columnar format for faster I/O).

## Notes about Parquet

- Parquet is a columnar, compressed binary format. It is preferred for large datasets because it reduces file size and makes column-subset reads faster.
- In pandas you can read/write Parquet as:

```python
import pandas as pd

# write
df.to_parquet('train/train.parquet', index=False)

# read
df = pd.read_parquet('train/train.parquet')
```

If you work on a machine without `pyarrow` or `fastparquet` installed, install one of them (for example `pip install pyarrow`) before reading/writing Parquet files.

## Running the pipeline (quick examples)

- Single-process run:

```bash
python src/run_preprocessing_pipeline.py
```

- MPI (multi-process) run — requires system MPI (OpenMPI/MPICH) and `mpi4py`:

```bash
mpirun -np 4 python src/run_preprocessing_pipeline.py --mpi
```

## Next steps / tips

- If you want a more detailed data dictionary (column meanings and derived features), I can generate `docs/data_dictionary.md` from the pipeline code and sample outputs.
- If you'd like sample notebooks showing how to load the Parquet outputs and inspect features, I can add one under `notebooks/`.

## How MPI is used in this project (practical details)

- Pattern: when you run with `--mpi` the runner uses `mpi4py` (`MPI.COMM_WORLD`) to split work between worker ranks. Each rank handles a portion of the overall task (file chunks or a slice of Monte Carlo runs) and then the root rank (usually `rank == 0`) aggregates results.

	- Typical code idioms you will find in `src/`:
		- `from mpi4py import MPI`
		- `comm = MPI.COMM_WORLD; rank = comm.Get_rank(); size = comm.Get_size()`
		- Partition data by index ranges or chunk ids: `start, end = compute_chunk(rank, size, n_items)`
		- Per-worker outputs to avoid write contention: `processed_data/part-{rank}.parquet` then root merges parts.
		- Aggregate with `comm.gather(...)` or `comm.reduce(...)` for small summaries; for large outputs prefer per-worker files + a final merge on the root.

	- Monte Carlo / randomness:
		- Seed per-worker RNG deterministically, for example `seed = base_seed + rank`, to make runs reproducible and independent across ranks.

	- Practical tips and requirements:
		- MPI must be installed on the system (OpenMPI or MPICH) and `mpi4py` must be installed in the Python environment.
		- Run with `mpirun -np N python src/run_preprocessing_pipeline.py --mpi` where `N` is the number of MPI processes.
		- On shared filesystems prefer per-rank temp files and a single final merge to avoid simultaneous writes to one file.
		- If native threads and MPI interact badly, set `OMP_NUM_THREADS=1` or bind processes to cores: `mpirun --bind-to core ...`.
		- For debugging run with a small `-np` (2 or 4) and check per-rank logs (files often named `worker-{rank}.log`).
x`
