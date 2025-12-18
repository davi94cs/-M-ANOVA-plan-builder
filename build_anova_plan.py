from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pandas as pd

# FAMILITY MAP 

DEFAULT_CLF_FAMILY_MAP: Dict[str, str] = {
    "CAT": "boosting", "HGB": "boosting", "XGB": "boosting",
    "RF": "bagging", "ET": "bagging",
    "DT": "tree",
    "LR": "linear",
    "LDA": "discriminant_linear",
    "GNB": "naive_bayes",
    "KNN": "knn",
    "MLP": "mlp",
    "SVC": "kernel_svm",
}

# LOGGING

def setup_logger(name: str = "anova_plan", level: str = "INFO", log_file: Optional[Union[str, Path]] = None) -> logging.Logger:

    """Configure console + optional file logging (idempotent)."""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",)
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
class PlanConfig:

    """
    Configuration for building the ANOVA plan.
    """

    max_hours_total: float
    n_dist: int = 1
    cost_dataset_size: int = 24000
    require_ex_time_sec: bool = False

# HELPERS

def parse_csv_list(s: Optional[str]) -> List[str]:
    """Parse comma-separated values into a list of stripped strings."""
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]

def load_family_map(use_default: bool, json_path: Optional[str], json_inline: Optional[str], logger: logging.Logger) -> Dict[str, str]:

    """
    Load classifier->family mapping.

    Priority:
      1) --family-map-json-inline
      2) --family-map-json path
      3) default map (if use_default True)
      4) empty map
    """

    if json_inline:
        try:
            m = json.loads(json_inline)
            if not isinstance(m, dict):
                raise ValueError("Inline family map must be a JSON object.")
            return {str(k): str(v) for k, v in m.items()}
        except Exception as e:
            raise ValueError(f"Invalid --family-map-json-inline: {e}") from e
    if json_path:
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"Family map JSON not found: {p}")
        m = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(m, dict):
            raise ValueError("Family map JSON must be an object {classifier: family}.")
        return {str(k): str(v) for k, v in m.items()}
    return dict(DEFAULT_CLF_FAMILY_MAP) if use_default else {}

def read_time_table(path: Union[str, Path], sep: str, dtype: Optional[str], logger: logging.Logger) -> pd.DataFrame:

    """Read the time table (TSV/CSV) robustly."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    logger.info("Reading time table: %s (sep=%r)", p, sep)
    return pd.read_csv(p, sep=sep, dtype=dtype)

def normalize_time_table(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:

    """
    Normalize basic columns (strip strings, ensure ints/floats).
    Requires at least: classifier, dataset_size and either ex_time_sec or ex_time_h.
    """

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required_base = {"classifier", "dataset_size"}
    missing = required_base - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["classifier"] = df["classifier"].astype(str).str.strip()
    df["dataset_size"] = pd.to_numeric(df["dataset_size"], errors="coerce").astype("Int64")
    if "fs_method" in df.columns:
        df["fs_method"] = df["fs_method"].astype(str).str.strip()
    bad_size = int(df["dataset_size"].isna().sum())
    if bad_size:
        logger.warning("Found %d rows with invalid dataset_size (NaN). They will be ignored in costing.", bad_size)
    # time columns
    if "ex_time_sec" in df.columns:
        df["ex_time_sec"] = pd.to_numeric(df["ex_time_sec"], errors="coerce")
    if "ex_time_h" in df.columns:
        df["ex_time_h"] = pd.to_numeric(df["ex_time_h"], errors="coerce")
    return df

def ensure_ex_time_sec(df: pd.DataFrame, require: bool, logger: logging.Logger) -> pd.DataFrame:
    """Ensure ex_time_sec exists; optionally compute from ex_time_h."""
    df = df.copy()
    if "ex_time_sec" in df.columns and df["ex_time_sec"].notna().any():
        return df
    if require:
        raise ValueError("ex_time_sec is required but not present/usable.")
    if "ex_time_h" not in df.columns:
        raise ValueError("Neither ex_time_sec nor ex_time_h is available to compute costs.")
    logger.info("ex_time_sec not available; computing ex_time_sec = ex_time_h * 3600.")
    df["ex_time_sec"] = df["ex_time_h"] * 3600.0
    return df


# SELECTION LOGIC

def optimize_classifiers_for_anova(df_time: pd.DataFrame, cfg: PlanConfig, clf_family_map: Optional[Dict[str, str]], logger: logging.Logger) -> Dict[str, object]:

    """
    Select a subset of classifiers under budget with one-model-per-family constraint.

    Cost definition:
      cost(clf) = sum(ex_time_sec) over rows with dataset_size == cfg.cost_dataset_size
      effective_cost = cost(clf) * cfg.n_dist

    Budget:
      sum(effective_cost) <= cfg.max_hours_total * 3600

    Objective:
      maximize #selected classifiers
      tie-breaker: minimize total cost

    Returns:
    dict with keys:
      - selected_classifiers: list[str]
      - total_time_sec: float (best cost under budget, already includes n_dist)
      - total_time_h: float
    """

    df = df_time.copy()
    df["classifier"] = df["classifier"].astype(str).str.strip()
    fam_map = clf_family_map or {}
    def family(clf: str) -> str:
        return fam_map.get(clf, clf)
    # cost per classifier at cost_dataset_size (single distribution)
    base = df[df["dataset_size"] == cfg.cost_dataset_size].copy()
    cost_per_clf = base.groupby("classifier", dropna=False)["ex_time_sec"].sum().to_dict()
    # multiply by n_dist
    cost_per_clf = {clf: float(cost) * float(cfg.n_dist) for clf, cost in cost_per_clf.items() if pd.notna(cost)}
    # family -> list[(clf, cost)]
    family_to_models: Dict[str, List[Tuple[str, float]]] = {}
    for clf, cost in cost_per_clf.items():
        fam = family(clf)
        family_to_models.setdefault(fam, []).append((clf, float(cost)))
    families = sorted(family_to_models.keys())
    max_sec = float(cfg.max_hours_total) * 3600.0
    best_selected: List[str] = []
    best_num = 0
    best_cost = float("inf")
    # Depth-first search on families (choose 0 or 1 model per family)
    def dfs(f_idx: int, cur_cost: float, cur_selected: List[str]) -> None:
        nonlocal best_selected, best_num, best_cost
        if cur_cost > max_sec:
            return
        if f_idx == len(families):
            cur_num = len(cur_selected)
            if (cur_num > best_num) or (cur_num == best_num and cur_cost < best_cost):
                best_num = cur_num
                best_cost = cur_cost
                best_selected = list(cur_selected)
            return
        fam = families[f_idx]
        models = family_to_models[fam]
        # Option 1: skip this family
        dfs(f_idx + 1, cur_cost, cur_selected)
        # Option 2: pick one model in this family
        # (small heuristic: try cheaper ones first -> earlier pruning)
        for clf, cost in sorted(models, key=lambda x: x[1]):
            cur_selected.append(clf)
            dfs(f_idx + 1, cur_cost + cost, cur_selected)
            cur_selected.pop()
    dfs(0, 0.0, [])
    return {"selected_classifiers": best_selected, "total_time_sec": 0.0 if best_num == 0 else float(best_cost), "total_time_h": 0.0 if best_num == 0 else float(best_cost) / 3600.0,
            "budget_time_sec": max_sec, "budget_time_h": float(cfg.max_hours_total), "n_selected": int(best_num), "n_families": int(len(families))}

def build_anova_plan(df_time: pd.DataFrame, cfg: PlanConfig, clf_family_map: Dict[str, str], logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, object]]:

    """
    Build ANOVA plan by selecting classifiers and marking rows.

    Returns:
    (df_plan, info)
      df_plan: dataframe with anova_selected flag (then typically filtered to True)
      info: selection summary and cost diagnostics
    """

    base_info = optimize_classifiers_for_anova(df_time=df_time, cfg=cfg, clf_family_map=clf_family_map, logger=logger)
    selected = set(base_info["selected_classifiers"])
    df_plan = df_time.copy()
    df_plan["anova_selected"] = df_plan["classifier"].isin(selected)
    # compute total cost from selected rows at cost_dataset_size
    mask_cost = (df_plan["dataset_size"] == cfg.cost_dataset_size) & (df_plan["anova_selected"])
    total_time_sec_selected_single = float(df_plan.loc[mask_cost, "ex_time_sec"].sum())
    total_time_sec_selected = total_time_sec_selected_single * float(cfg.n_dist)
    info = dict(base_info)
    info.update({"n_dist": int(cfg.n_dist), "cost_dataset_size": int(cfg.cost_dataset_size), "total_time_sec_selected_single": total_time_sec_selected_single,
                 "total_time_sec_selected": total_time_sec_selected, "total_time_h_selected": total_time_sec_selected / 3600.0})
    return df_plan, info

def replicate_plan_rows(df_plan_selected: pd.DataFrame, n_dist: int, group_cols: Optional[List[str]] = None, logger: Optional[logging.Logger] = None) -> pd.DataFrame:

    """
    Replicate each row n_dist times and add a 'distribution' column (1..n_dist) per group.

    group_cols default: ['fs_method','classifier','dataset_size'] if present, else ['classifier','dataset_size'].
    """

    if n_dist <= 1:
        return df_plan_selected
    df = df_plan_selected.loc[df_plan_selected.index.repeat(n_dist)].copy()
    if group_cols is None:
        group_cols = []
        for c in ["fs_method", "classifier", "dataset_size"]:
            if c in df.columns:
                group_cols.append(c)
        if not group_cols:
            group_cols = ["classifier"]
    df["distribution"] = df.groupby(group_cols, dropna=False).cumcount() + 1
    if logger:
        logger.info("Replicated rows by n_dist=%d -> %d rows.", n_dist, len(df))
    return df

# OUTPUT HELPERS

def write_table(df: pd.DataFrame, path: Union[str, Path], sep: str, logger: logging.Logger) -> None:
    """Write TSV/CSV output."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing output: %s (sep=%r)", p, sep)
    df.to_csv(p, sep=sep, index=False)

def format_selected_with_families(selected: Sequence[str], fam_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return list of (clf, family) pairs for logging/reporting."""
    return [(clf, fam_map.get(clf, clf)) for clf in selected]

# CLI

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build an ANOVA plan under a time budget (one model per family).")
    p.add_argument("--input", "-i", required=True, help="Input time table path (TSV/CSV).")
    p.add_argument("--sep", default="\t", help="Input separator (default: tab).")
    p.add_argument("--output", "-o", required=True, help="Output plan path (TSV).")
    p.add_argument("--output-sep", default="\t", help="Output separator (default: tab).")
    p.add_argument("--max-hours", type=float, required=True, help="Max total hours budget.")
    p.add_argument("--n-dist", type=int, default=1, help="Number of distributions/replications (default: 1).")
    p.add_argument("--cost-dataset-size", type=int, default=24000, help="Dataset size used for costing (default: 24000).")
    p.add_argument("--exclude-fs", default="", help="Comma-separated fs_method values to exclude (e.g. 'SKB').")
    p.add_argument("--exclude-classifiers", default="", help="Comma-separated classifiers to exclude (e.g. 'ADA,QDA').")
    p.add_argument("--use-default-family-map", action="store_true",
                   help="Use built-in DEFAULT_CLF_FAMILY_MAP (recommended).")
    p.add_argument("--family-map-json", default=None,
                   help="Path to JSON mapping {classifier: family}. Overrides default if provided.")
    p.add_argument("--family-map-json-inline", default=None,
                   help="Inline JSON mapping {classifier: family}. Overrides everything.")
    p.add_argument("--replicate-rows", action="store_true",
                   help="If set and n-dist>1, replicate rows and add 'distribution' column.")
    p.add_argument("--drop-columns", default="classifier_clean,anova_selected,ex_time_sec,is_estimated,alpha,split",
                   help="Comma-separated columns to drop in final output (best-effort).")
    p.add_argument("--keep-only-selected", action="store_true",
                   help="If set, keep only selected rows (anova_selected==True). Default: True behavior is applied anyway.")
    p.add_argument("--no-keep-only-selected", dest="keep_only_selected", action="store_false",
                   help="If set, DO NOT filter to selected rows (keeps full table with anova_selected flag).")
    p.set_defaults(keep_only_selected=True)
    p.add_argument("--dtype", default="infer", help="Read dtype: 'infer' or 'str' (default: infer).")
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO).")
    p.add_argument("--log-file", default=None, help="Optional log file path.")
    return p

# MAIN

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logger = setup_logger(level=args.log_level, log_file=args.log_file)
    t0 = time.time()
    try:
        dtype = None if args.dtype == "infer" else args.dtype
        fam_map = load_family_map(use_default=args.use_default_family_map, json_path=args.family_map_json, json_inline=args.family_map_json_inline, logger=logger)
        df = read_time_table(args.input, sep=args.sep, dtype=dtype, logger=logger)
        df = normalize_time_table(df, logger=logger)
        df = ensure_ex_time_sec(df, require=False, logger=logger)
        # Filters
        exclude_fs = set(parse_csv_list(args.exclude_fs))
        exclude_clf = set(parse_csv_list(args.exclude_classifiers))
        if exclude_fs and "fs_method" in df.columns:
            before = len(df)
            df = df[~df["fs_method"].isin(exclude_fs)].copy()
            logger.info("Excluded fs_method=%s: %d -> %d rows", sorted(exclude_fs), before, len(df))
        if exclude_clf:
            before = len(df)
            df = df[~df["classifier"].isin(exclude_clf)].copy()
            logger.info("Excluded classifiers=%s: %d -> %d rows", sorted(exclude_clf), before, len(df))
        cfg = PlanConfig(max_hours_total=float(args.max_hours), n_dist=int(args.n_dist), cost_dataset_size=int(args.cost_dataset_size), require_ex_time_sec=False)
        df_plan, info = build_anova_plan(df_time=df, cfg=cfg, clf_family_map=fam_map, logger=logger)
        selected = info["selected_classifiers"]
        selected_with_family = format_selected_with_families(selected, fam_map)
        logger.info("n_dist: %d", info["n_dist"])
        logger.info("Selected classifiers (%d): %s", len(selected), selected)
        logger.info("Selected models (model, family):")
        for clf, fam in selected_with_family:
            logger.info("  - (%s, %s)", clf, fam)
        logger.info("Cost (h, DFS, includes n_dist): %.4f / Threshold (h): %.2f",
                    float(info["total_time_h"]), float(args.max_hours))
        logger.info("Cost (h, cost_dataset_size selected * n_dist): %.4f",
                    float(info["total_time_h_selected"]))
        # Output shaping
        if args.keep_only_selected:
            df_out = df_plan[df_plan["anova_selected"] == True].copy()
        else:
            df_out = df_plan.copy()
        # Drop columns (best-effort)
        drop_cols = [c.strip() for c in str(args.drop_columns).split(",") if c.strip()]
        df_out = df_out.drop(columns=drop_cols, errors="ignore")
        # Replicate rows if requested
        if args.replicate_rows:
            df_out = replicate_plan_rows(df_out, n_dist=cfg.n_dist, logger=logger)
        # Optional: cast non-numeric columns to string (safer exports)
        # Keep ex_time_h numeric if present
        numeric_keep = {"ex_time_h", "ex_time_sec", "dataset_size", "distribution"}
        cast_cols = [c for c in df_out.columns if c not in numeric_keep]
        for c in cast_cols:
            df_out[c] = df_out[c].astype("string")
        write_table(df_out, args.output, sep=args.output_sep, logger=logger)
        t1 = time.time()
        logger.info("Runtime (sec): %.4f", t1 - t0)
        return 0
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

# python3 build_anova_plan.py -i input/time_table.tsv --sep $'\t' -o output/100h/anova_plan_100h.tsv --output-sep $'\t' --max-hours 100 --n-dist 3 --cost-dataset-size 24000 --exclude-fs SKB --exclude-classifiers ADA,QDA --use-default-family-map --replicate-rows --log-file output/100h/log_100h.txt

