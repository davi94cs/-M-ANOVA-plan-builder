from __future__ import annotations
import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

# LOGGING

def setup_logger(name: str = "time_table", level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> logging.Logger:

    """
    Configure and return a logger with both console and optional file handlers.
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    # Avoid duplicate handlers in repeated invocations (e.g., notebooks/tests)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# CONFIG

@dataclass(frozen=True)
class TimeTableConfig:

    """
    Configuration for building the time table.
    """
    target_sizes: Tuple[int, ...] = (500, 3000, 24000)
    base_size: int = 3000
    alpha_min: float = 1.0
    alpha_max: float = 3.0
    ceil_decimals: int = 1
    alpha_ratio_threshold: float = 1.05

# HELPERS

def ceil_n(x: float, n: int = 1) -> float:

    """
    Ceiling-round a number to the n-th decimal digit.
    """

    if pd.isna(x):
        return x
    factor = 10 ** n
    return math.ceil(float(x) * factor) / factor

def parse_ex_time(val: object, ceil_decimals: int = 1) -> float:

    """
    Parse an execution-time value into seconds.

    Supported input formats:
      - NaN/None -> NaN
      - numeric -> seconds
      - string float: "123.4"
      - compound duration: "Xd Yh Zm Ws" (spaces optional), e.g. "1h 2m 3s", "2d3h", "10m"
    """

    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return ceil_n(float(val), ceil_decimals)
    s = str(val).strip()
    if not s:
        return np.nan
    # Try plain float
    try:
        return ceil_n(float(s), ceil_decimals)
    except ValueError:
        pass
    # Parse "Xd Yh Zm Ws" (spaces optional)
    total_sec = 0.0
    parts = (s.replace("d", " d").replace("h", " h").replace("m", " m").replace("s", " s").split())
    val_buffer: Optional[str] = None
    for token in parts:
        token = token.strip()
        if not token:
            continue
        if token in ("d", "h", "m", "s"):
            if val_buffer is None:
                continue
            try:
                v = float(val_buffer)
            except ValueError:
                return np.nan
            if token == "d":
                total_sec += v * 86400
            elif token == "h":
                total_sec += v * 3600
            elif token == "m":
                total_sec += v * 60
            elif token == "s":
                total_sec += v
            val_buffer = None
        else:
            val_buffer = token
    # Trailing number without unit -> interpret as seconds
    if val_buffer is not None:
        try:
            total_sec += float(val_buffer)
        except ValueError:
            return np.nan
    return ceil_n(total_sec, ceil_decimals)

def ensure_base_in_targets(target_sizes: Sequence[int], base_size: int) -> Tuple[int, ...]:

    """Ensure base_size is included in target_sizes (sorted, unique)."""

    return tuple(sorted(set(map(int, target_sizes)) | {int(base_size)}))

def sample_distribution_from_split(split_val: object) -> str:

    """
    Convert a split percentage (e.g. 80.0) into a label "80_20".
    Returns "unknown" if not parseable.
    """

    try:
        p = float(split_val)
        return f"{int(p)}_{int(100 - p)}"
    except Exception:
        return "unknown"

# CORE LOGIC

def normalize_input_df(df_raw: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:

    """
    Normalize input DataFrame columns and types.

    Expected columns (case-insensitive, extra columns allowed):
      - FEATURE_SEL -> fs_method
      - CLASSIFIER  -> classifier
      - SPLITTING   -> split
      - EX_TIME     -> ex_time
    Optional:
      - KFOLD (will be dropped if present)

    Returns pd.DataFrame --> Normalized DataFrame with columns: fs_method, classifier, split, ex_time
    """

    df = df_raw.copy()
    # Keep first 5 columns behavior from original code, but safer:
    # if df has < 5 columns, keep all.
    df = df.iloc[:, : min(5, df.shape[1])]
    # Drop KFOLD if present
    df = df.drop(columns=["KFOLD"], errors="ignore")
    # Strip column names and standardize
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {"FEATURE_SEL": "fs_method", "CLASSIFIER": "classifier", "SPLITTING": "split", "EX_TIME": "ex_time"}
    df = df.rename(columns=rename_map)
    required = {"fs_method", "classifier", "split", "ex_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")
    # Split numeric extraction (e.g. "80_20" -> 80)
    split_num = df["split"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
    df["split"] = pd.to_numeric(split_num, errors="coerce")
    bad_split = df["split"].isna().sum()
    if bad_split:
        logger.warning("Found %d rows with unparseable 'split' values (set to NaN).", bad_split)
    return df

def estimate_alpha(df: pd.DataFrame, cfg: TimeTableConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, float]:

    """
    Estimate local alpha per (fs_method, classifier) from mean T80/T70.

    alpha = log(T80/T70) / log((0.8)/(0.7))
    alpha is clamped to [alpha_min, alpha_max] and ceiling-rounded.
    """

    ratio_N = 0.8 / 0.7
    log_ratio_N = math.log(ratio_N)
    rows = []
    grouped = df.groupby(["fs_method", "classifier"], dropna=False)
    for (fs, clf), g in grouped:
        splits = set(g["split"].dropna().unique())
        if not ({70.0, 80.0} <= splits):
            continue
        t70 = g.loc[g["split"] == 70.0, "ex_time_sec"].mean()
        t80 = g.loc[g["split"] == 80.0, "ex_time_sec"].mean()
        if not (t70 > 0 and t80 > 0):
            continue
        ratio_T = t80 / t70
        if ratio_T <= cfg.alpha_ratio_threshold:
            continue
        alpha_local = math.log(ratio_T) / log_ratio_N
        alpha_local = max(cfg.alpha_min, min(cfg.alpha_max, alpha_local))
        alpha_local = ceil_n(alpha_local, cfg.ceil_decimals)
        rows.append({"fs_method": fs, "classifier": clf, "alpha_local": alpha_local})
    df_alpha = pd.DataFrame(rows)
    if not df_alpha.empty:
        alpha_med = ceil_n(float(df_alpha["alpha_local"].median()), cfg.ceil_decimals)
        logger.info("Estimated alpha for %d (fs_method, classifier) pairs. Median alpha=%.3f", len(df_alpha), alpha_med)
    else:
        alpha_med = ceil_n(1.0, cfg.ceil_decimals)
        logger.warning("No valid alpha estimates found; falling back to alpha=%.3f", alpha_med)
    return df_alpha, float(alpha_med)

def build_time_table(df_raw: pd.DataFrame, cfg: TimeTableConfig, logger: logging.Logger) -> pd.DataFrame:

    """
    Build the time table for all target dataset sizes.
    """

    target_sizes = ensure_base_in_targets(cfg.target_sizes, cfg.base_size)
    df = normalize_input_df(df_raw, logger=logger)
    # Parse execution time to seconds
    df["ex_time_sec"] = df["ex_time"].apply(lambda v: parse_ex_time(v, ceil_decimals=cfg.ceil_decimals))
    bad_time = df["ex_time_sec"].isna().sum()
    if bad_time:
        logger.warning("Found %d rows with unparseable 'ex_time' (set to NaN).", bad_time)
    # Estimate alpha
    df_alpha, alpha_med = estimate_alpha(df, cfg=cfg, logger=logger)
    # Aggregate base runtimes per split
    agg = (df.groupby(["fs_method", "classifier", "split"], dropna=False).agg(ex_time_base_sec=("ex_time_sec", "mean")).reset_index())
    agg["sample_distribution"] = agg["split"].apply(sample_distribution_from_split)
    # Merge alpha per (fs, clf)
    if not df_alpha.empty:
        agg = agg.merge(df_alpha, on=["fs_method", "classifier"], how="left")
    else:
        agg["alpha_local"] = np.nan
    # Choose alpha: local if available, else global median
    agg["alpha"] = agg["alpha_local"].apply(lambda a: ceil_n(float(a), cfg.ceil_decimals) if not pd.isna(a) else ceil_n(alpha_med, cfg.ceil_decimals))
    # Generate rows for each target size
    rows = []
    for _, r in agg.iterrows():
        if pd.isna(r["ex_time_base_sec"]) or r["ex_time_base_sec"] <= 0:
            continue
        T_base = float(r["ex_time_base_sec"])
        alpha_val = float(r["alpha"])
        base_info = {"fs_method": r["fs_method"], "classifier": r["classifier"], "sample_distribution": r["sample_distribution"], "alpha": alpha_val}
        for N in target_sizes:
            ratio = float(N) / float(cfg.base_size)
            ex_sec = T_base if N == cfg.base_size else T_base * (ratio ** alpha_val)
            rows.append({**base_info, "dataset_size": int(N), "ex_time_sec": ex_sec, "ex_time_h": ex_sec / 3600.0})
    out = pd.DataFrame(rows)
    # Ceiling-round all float columns
    if not out.empty:
        float_cols = out.select_dtypes(include=["float", "float64", "float32"]).columns
        for c in float_cols:
            out[c] = out[c].apply(lambda x: ceil_n(x, cfg.ceil_decimals))
    logger.info("Built time table with %d rows.", len(out))
    return out

# CLI 

def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    p = argparse.ArgumentParser(description="Build an execution-time scaling table from base observed runtimes.")
    p.add_argument("--input", "-i", required=True, help="Path to input TSV/CSV file.")
    p.add_argument("--sep", default="\t", help="Input separator (default: tab). Use ',' for CSV.")
    p.add_argument("--output", "-o", required=True, help="Path to output TSV file.")
    p.add_argument("--output-sep", default="\t", help="Output separator (default: tab).")
    p.add_argument("--target-sizes", default="500,3000,24000", help="Comma-separated dataset sizes (e.g. '500,3000,24000').")
    p.add_argument("--base-size", type=int, default=3000, help="Base dataset size (default: 3000).")
    p.add_argument("--alpha-min", type=float, default=1.0, help="Minimum alpha (default: 1.0).")
    p.add_argument("--alpha-max", type=float, default=3.0, help="Maximum alpha (default: 3.0).")
    p.add_argument("--alpha-ratio-threshold", type=float, default=1.05, help="Min T80/T70 ratio to accept alpha estimate (default: 1.05).")
    p.add_argument("--ceil-decimals", type=int, default=1, help="Decimals for ceiling rounding (default: 1).")
    p.add_argument("--filter-sample-distribution", default=None, help="Optional filter, e.g. '80_20' (keeps only that split distribution).")
    p.add_argument("--drop-sample-distribution", action="store_true", help="Drop 'sample_distribution' column in output.")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    p.add_argument("--log-file", default=None, help="Optional log file path.")
    return p

def parse_target_sizes(s: str) -> Tuple[int, ...]:

    """Parse comma-separated sizes into a tuple of ints."""

    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --target-sizes.")
    sizes = []
    for p in parts:
        try:
            sizes.append(int(p))
        except ValueError as e:
            raise ValueError(f"Invalid target size '{p}'. Must be int.") from e
    return tuple(sizes)

# PROCESS TABLE

def read_table(path: Union[str, Path], sep: str, logger: logging.Logger) -> pd.DataFrame:

    """
    Read input table safely (keeps strings, like original code).

    Returns
    -------
    pd.DataFrame
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info("Reading input: %s (sep=%r)", path, sep)
    return pd.read_csv(path, sep=sep, dtype=str)

def write_table(df: pd.DataFrame, path: Union[str, Path], sep: str, logger: logging.Logger) -> None:

    """Write output table to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing output: %s (sep=%r)", path, sep)
    df.to_csv(path, sep=sep, index=False)

# MAIN

def main(argv: Optional[Sequence[str]] = None) -> int:

    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logger = setup_logger(level=args.log_level, log_file=args.log_file)
    try:
        target_sizes = parse_target_sizes(args.target_sizes)
        cfg = TimeTableConfig(target_sizes=target_sizes, base_size=args.base_size, alpha_min=args.alpha_min, alpha_max=args.alpha_max, ceil_decimals=args.ceil_decimals,
                              alpha_ratio_threshold=args.alpha_ratio_threshold)
        df_in = read_table(args.input, sep=args.sep, logger=logger)
        out = build_time_table(df_in, cfg=cfg, logger=logger)
        if args.filter_sample_distribution is not None:
            sd = args.filter_sample_distribution.strip()
            before = len(out)
            out = out[out["sample_distribution"] == sd].copy()
            logger.info("Filtered sample_distribution=%s: %d -> %d rows", sd, before, len(out))
        if args.drop_sample_distribution and "sample_distribution" in out.columns:
            out = out.drop(columns=["sample_distribution"])
        write_table(out, args.output, sep=args.output_sep, logger=logger)
        logger.info("Done.")
        return 0
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

# python3 build_time_table.py --input input/merged.tsv --sep $'\t' --output output/time_table.tsv --output-sep $'\t' --target-sizes 500,3000,24000 --base-size 3000 --filter-sample-distribution 80_20 --drop-sample-distribution --log-level INFO

