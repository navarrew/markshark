#!/usr/bin/env python3
r"""
MarkShark
stats_tools.py
Append item statistics to a multiple-choice results CSV produced by score_core.py

Key features
------------
- **KEY row with letters**: Auto-detects the key row by default (looks for a cell == "KEY"
  in any non-item column, case-insensitive). If not found, defaults to the first data row
  (row index 0 after the header). You can override with --key-row-index.
- Builds a correctness matrix (0/1) by comparing student letters to the key.
- Appends two rows to the bottom: "Pct correct" and "Point–biserial".
- Computes exam reliability: KR-20 / KR-21.
- Optional: per-item distractor analysis CSV (option counts/proportions and option-biserial).
- Optional: nonparametric item characteristic plots.

CLI flags cover item naming, key row index (or auto), output paths, and plots.
"""
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

# Matplotlib is only used if --plots-dir is provided
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def detect_item_columns(df: pd.DataFrame, pattern: str) -> List[str]:
    regex = re.compile(pattern)
    return [c for c in df.columns if isinstance(c, str) and regex.fullmatch(c.strip())]


def detect_key_row_index(df: pd.DataFrame,
                         item_cols: List[str],
                         key_label: str = "KEY") -> int:
    """
    Try to locate a row that contains the string 'KEY' (case-insensitive) in any
    non-item column. If found, return that row index; else return 0 (first data row).
    """
    non_item_cols = [c for c in df.columns if c not in item_cols]
    label_norm = str(key_label).strip().lower()
    if non_item_cols:
        for i in range(len(df)):
            row_vals = df.loc[df.index[i], non_item_cols]
            # any non-item cell equals 'KEY' (case-insensitive)?
            if row_vals.astype(str).str.strip().str.lower().eq(label_norm).any():
                return i
    else:
        # If every column is an item, search entire row for 'KEY'
        for i in range(len(df)):
            if df.iloc[i].astype(str).str.strip().str.lower().eq(label_norm).any():
                return i
    # Fallback
    return 0


def prepare_correctness_matrix(df: pd.DataFrame,
                               item_cols: List[str],
                               key_row_index: int,
                               answers_mode: str = "auto"
                               ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    r"""
    Returns:
      items_num: DataFrame (students x items) of 0/1/NaN correctness
      total_scores: Series of total correct per student (NaN treated as 0 for totals)
      students_df: original df with the KEY row removed
      key_series: Series with key letter for each item (uppercased string; NaN if unavailable)
    """
    if key_row_index < 0 or key_row_index >= len(df):
        raise SystemExit(f"--key-row-index={key_row_index} is out of bounds for {len(df)} rows.")

    key_row = df.iloc[key_row_index]
    students_df = df.drop(index=df.index[key_row_index]).reset_index(drop=True)

    # Mode selection
    if answers_mode not in {"auto", "letters", "binary"}:
        raise SystemExit("--answers-mode must be one of: auto, letters, binary")

    def has_letters(series: pd.Series) -> bool:
        vals = series.dropna().astype(str).str.strip()
        if vals.empty:
            return False
        return vals.str.fullmatch(r"[A-Za-z]+").any()

    if answers_mode == "auto":
        key_has_letters = any(has_letters(pd.Series([key_row[c]])) for c in item_cols)
        student_has_letters = any(has_letters(students_df[c]) for c in item_cols)
        mode = "letters" if (key_has_letters or student_has_letters) else "binary"
    else:
        mode = answers_mode

    if mode == "binary":
        items_num = students_df[item_cols].apply(pd.to_numeric, errors="coerce")
        key_series = pd.Series({c: np.nan for c in item_cols})
    else:
        key_letters = key_row[item_cols].astype(str).str.strip().str.upper()
        items_num = pd.DataFrame(index=students_df.index, columns=item_cols, dtype=float)
        for c in item_cols:
            key_val = key_letters[c]
            stud_vals = students_df[c].astype(str).str.strip().str.upper()
            if (key_val is None) or (key_val == "") or (not re.fullmatch(r"[A-Z]", key_val)):
                items_num[c] = np.nan
                key_letters[c] = np.nan
            else:
                eq = (stud_vals == key_val)
                is_blank = students_df[c].isna() | (students_df[c].astype(str).str.strip() == "")
                col = pd.Series(np.where(is_blank.to_numpy(), np.nan, np.where(eq.to_numpy(), 1.0, 0.0)),
                                index=students_df.index)
                items_num[c] = col
        key_series = key_letters

    total_scores = items_num.fillna(0).sum(axis=1)
    return items_num, total_scores, students_df, key_series


def point_biserial(item: pd.Series, total_minus_item: pd.Series) -> float:
    x = pd.to_numeric(item, errors="coerce")
    y = pd.to_numeric(total_minus_item, errors="coerce")
    mask = y.notna()
    x = x[mask].fillna(0)
    y = y[mask]
    p = x.mean()
    q = 1 - p
    if p == 0 or p == 1:
        return float("nan")
    y1 = y[x == 1]
    y0 = y[x == 0]
    if len(y1) == 0 or len(y0) == 0:
        return float("nan")
    M1 = y1.mean(); M0 = y0.mean()
    s = y.std(ddof=1)
    if s == 0 or np.isnan(s):
        return float("nan")
    return float(((M1 - M0) / s) * np.sqrt(p * q))


def kr20(items_num: pd.DataFrame, total_scores: pd.Series) -> float:
    k = items_num.shape[1]
    if k < 2:
        return float("nan")
    p = items_num.mean(axis=0)  # ignore NaN
    pq_sum = (p * (1 - p)).sum()
    var_total = np.var(total_scores, ddof=1)
    if var_total <= 0 or np.isnan(var_total):
        return float("nan")
    return float((k / (k - 1.0)) * (1.0 - (pq_sum / var_total)))


def kr21(items_num: pd.DataFrame, total_scores: pd.Series) -> float:
    k = items_num.shape[1]
    if k < 2:
        return float("nan")
    p_bar = items_num.mean(axis=0).mean()
    var_total = np.var(total_scores, ddof=1)
    if var_total <= 0 or np.isnan(var_total):
        return float("nan")
    return float((k / (k - 1.0)) * (1.0 - (k * p_bar * (1 - p_bar)) / var_total))


def make_summary_row(df: pd.DataFrame, item_cols: List[str], label: str, values: Dict[str, Any], label_col: Optional[str]) -> pd.Series:
    row = pd.Series({c: "" for c in df.columns})
    non_item_cols = [c for c in df.columns if c not in item_cols]
    if label_col is None:
        label_col = non_item_cols[0] if non_item_cols else df.columns[0]
    row[label_col] = label
    for c in item_cols:
        row[c] = values.get(c, np.nan)
    return row


def option_biserial(option_selected: pd.Series, total_minus_item: pd.Series) -> float:
    """Correlation between selecting a particular option (1/0) and total-minus-item."""
    x = pd.to_numeric(option_selected, errors="coerce").fillna(0)
    y = pd.to_numeric(total_minus_item, errors="coerce")
    mask = y.notna()
    x = x[mask]
    y = y[mask]
    p = x.mean()
    q = 1 - p
    if p == 0 or p == 1:
        return float("nan")
    y1 = y[x == 1]; y0 = y[x == 0]
    if len(y1) == 0 or len(y0) == 0:
        return float("nan")
    M1 = y1.mean(); M0 = y0.mean()
    s = y.std(ddof=1)
    if s == 0 or np.isnan(s):
        return float("nan")
    return float(((M1 - M0) / s) * np.sqrt(p * q))


def per_item_analysis(students_df: pd.DataFrame,
                      item_cols: List[str],
                      key_series: pd.Series,
                      total_scores: pd.Series) -> pd.DataFrame:
    records = []
    for c in item_cols:
        key = key_series.get(c, np.nan)
        choices = students_df[c].astype(str).str.strip().str.upper().replace({"": np.nan})
        opts = set(choices.dropna().unique().tolist())
        if isinstance(key, str) and re.fullmatch(r"[A-Z]", key):
            opts.add(key)
        opts = sorted([o for o in opts if isinstance(o, str) and re.fullmatch(r"[A-Z]", o)])

        correct = (choices == key).astype(float)
        correct[choices.isna()] = np.nan
        total_minus = total_scores - correct.fillna(0)

        p_correct = float(np.nanmean(correct)) if np.isfinite(np.nanmean(correct)) else np.nan
        pb_item = point_biserial(correct, total_minus)

        n_students = (~choices.isna()).sum()
        for o in opts:
            sel = (choices == o).astype(float)
            sel[choices.isna()] = np.nan
            count = int(np.nansum(sel))
            prop = (count / n_students) if n_students > 0 else np.nan
            ob = option_biserial(sel, total_minus)
            records.append({
                "item": c,
                "key": key if isinstance(key, str) else np.nan,
                "option": o,
                "is_key": bool(o == key),
                "option_count": count,
                "option_prop": prop,
                "option_biserial": ob,
                "item_difficulty": p_correct,
                "item_point_biserial": pb_item,
            })

        blanks = choices.isna().sum()
        if blanks > 0:
            sel_blank = choices.isna().astype(float)
            ob_blank = option_biserial(sel_blank, total_minus)
            records.append({
                "item": c,
                "key": key if isinstance(key, str) else np.nan,
                "option": "BLANK",
                "is_key": False,
                "option_count": int(blanks),
                "option_prop": (blanks / (n_students + blanks)) if (n_students + blanks) > 0 else np.nan,
                "option_biserial": ob_blank,
                "item_difficulty": p_correct,
                "item_point_biserial": pb_item,
            })

    return pd.DataFrame.from_records(records)


def save_exam_stats_csv(path: str, k: int, mean_total: float, sd_total: float, var_total: float,
                        avg_difficulty: float, kr20_val: float, kr21_val: float) -> None:
    out = pd.DataFrame([{
        "k_items": k,
        "mean_total": mean_total,
        "sd_total": sd_total,
        "var_total": var_total,
        "avg_difficulty": avg_difficulty,
        "KR20": kr20_val,
        "KR21": kr21_val,
    }])
    out.to_csv(path, index=False, float_format="%.3f", encoding="utf-8-sig")


def plot_item_characteristic(item_name: str,
                             item_series: pd.Series,
                             total_minus_item: pd.Series,
                             outdir: Path,
                             bins: int = 10) -> None:
    if plt is None:
        return
    x = pd.to_numeric(total_minus_item, errors="coerce")
    y = pd.to_numeric(item_series, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 5:
        return
    cats = pd.qcut(x, q=min(bins, len(x)), duplicates='drop')
    grp = pd.DataFrame({"x": x, "y": y}).groupby(cats, observed=True)
    prop = grp["y"].mean(); centers = grp["x"].mean()
    fig = plt.figure()
    plt.plot(centers.values, prop.values, marker='o')
    plt.xlabel("Total-minus-item score (bin center)")
    plt.ylabel("Proportion correct")
    plt.title(f"Item characteristic (nonparametric): {item_name}")
    outpath = outdir / f"{item_name}_icc.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)


def run(input_csv: str,
        output_csv: str,
        item_pattern: str,
        percent: bool,
        label_col: Optional[str],
        exam_stats_csv: Optional[str],
        plots_dir: Optional[str],
        key_row_index: Optional[int],
        answers_mode: str,
        item_report_csv: Optional[str],
        key_label: str,
        decimals: int = 3) -> None:
    """
    Compute exam and item statistics from results CSV.

    decimals
        Number of decimal places to round numeric outputs in all CSVs (default 3).
    """

    df = pd.read_csv(input_csv)
    item_cols = detect_item_columns(df, item_pattern)
    if not item_cols:
        raise SystemExit(f"No item columns matched pattern {item_pattern!r}. "
                         f"Available columns: {list(df.columns)}")

    # Auto-detect key row if not provided
    if key_row_index is None or key_row_index < 0:
        key_row_index = detect_key_row_index(df, item_cols, key_label=key_label)

    # Build correctness matrix
    items_num, total, students_df, key_series = prepare_correctness_matrix(
        df=df,
        item_cols=item_cols,
        key_row_index=key_row_index,
        answers_mode=answers_mode
    )

    # Difficulty & PB
    difficulty = items_num.mean(axis=0)
    pb_vals: Dict[str, float] = {}
    for col in item_cols:
        item_series = items_num[col]
        total_minus = total - item_series.fillna(0)
        pb_vals[col] = point_biserial(item_series, total_minus)

    # Summary rows appended under ORIGINAL dataframe (with key row still present)
    diff_values = {c: (float(difficulty[c]) * (100.0 if percent else 1.0)) if pd.notna(difficulty[c]) else np.nan
                   for c in item_cols}
    diff_label = "Pct correct (0-100)" if percent else "Pct correct (0-1)"
    pb_values = {c: (float(pb_vals[c]) if pd.notna(pb_vals[c]) else np.nan) for c in item_cols}
    diff_row = make_summary_row(df, item_cols, diff_label, diff_values, label_col)
    pb_row = make_summary_row(df, item_cols, "Point–biserial", pb_values, label_col)

    # Format summary rows' item cells to N-decimal strings so they look rounded in CSV
    def _format_row_ndp(row_series, cols, ndp: int):
        fmt = f"{{:.{ndp}f}}"
        for _c in cols:
            v = row_series.get(_c, None)
            try:
                if v is not None and v == v:  # not NaN
                    row_series[_c] = fmt.format(float(v))
            except Exception:
                pass
        return row_series

    diff_row = _format_row_ndp(diff_row, item_cols, decimals)
    pb_row  = _format_row_ndp(pb_row,  item_cols, decimals)

    df_out = pd.concat([df, pd.DataFrame([diff_row, pb_row])], ignore_index=True)

    # Apply rounding to numeric columns before writing (keeps numbers numeric wherever possible)
    # (Note: summary rows have strings already, which is fine for presentation.)
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].round(decimals)

    float_fmt = f"%.{decimals}f"
    df_out.to_csv(output_csv, index=False, float_format=float_fmt, encoding="utf-8-sig")

    # Exam-level reliability
    k = len(item_cols)
    mean_total = round(float(total.mean()), decimals)
    var_total  = round(float(total.var(ddof=1)), decimals)
    sd_total   = round(float(np.sqrt(total.var(ddof=1))) if total.var(ddof=1) >= 0 else float("nan"), decimals)
    avg_difficulty = round(float(difficulty.mean()), decimals)
    kr20_val = round(kr20(items_num, total), decimals)
    kr21_val = round(kr21(items_num, total), decimals)

    if exam_stats_csv:
        # If save_exam_stats_csv writes raw values, this keeps its outputs rounded.
        save_exam_stats_csv(exam_stats_csv, k, mean_total, sd_total, var_total, avg_difficulty, kr20_val, kr21_val)

    # Per-item distractor analysis (optional)
    if item_report_csv:
        report_df = per_item_analysis(students_df, item_cols, key_series, total)
        cols = ["item", "key", "option", "is_key",
                "option_count", "option_prop", "option_biserial",
                "item_difficulty", "item_point_biserial"]
        report_df = report_df[cols]

        # Round numeric columns
        num_cols_rep = report_df.select_dtypes(include=[np.number]).columns
        report_df[num_cols_rep] = report_df[num_cols_rep].round(decimals)
        report_df.to_csv(item_report_csv, index=False, float_format=float_fmt, encoding="utf-8-sig")

    # Optional plots (no change; if you annotate values on figures, round there too)
    if plots_dir:
        outdir = Path(plots_dir); outdir.mkdir(parents=True, exist_ok=True)
        for col in item_cols:
            item_series = items_num[col]
            total_minus = total - item_series.fillna(0)
            plot_item_characteristic(col, item_series, total_minus, outdir)


def main():
    ap = argparse.ArgumentParser(description="Append item difficulty/point–biserial, compute KR-20/KR-21, KEY auto-detect, and distractor analysis.")
    ap.add_argument("-i", "--input", required=True, help="Input CSV path.")
    ap.add_argument("-o", "--output", required=False, help="Output CSV path (default: <input> with _with_item_stats)." )
    ap.add_argument("--item-pattern", default=r"Q\d+", help=r"Regex for item columns (default: Q\\d+)." )
    ap.add_argument("--percent", action="store_true", help="Write difficulty as percent (0–100) instead of proportion (0–1)." )
    ap.add_argument("--label-col", default=None, help="Column where to place the summary row labels. Default: first non-item column (or first column)." )
    ap.add_argument("--exam-stats-csv", default=None, help="Optional path to write exam-level stats (k, mean, sd, KR-20, KR-21)." )
    ap.add_argument("--plots-dir", default=None, help="If set, writes simple item characteristic plots (PNG) to this directory." )
    ap.add_argument("--key-row-index", type=int, default=-1, help="Row index of the KEY answers (0-based). Default: -1 = auto-detect." )
    ap.add_argument("--answers-mode", choices=["auto", "letters", "binary"], default="auto",
                    help="Interpretation of item cells. 'auto' detects letters vs 0/1 (default)." )
    ap.add_argument("--item-report-csv", default=None, help="Optional path to write per-item distractor analysis (long table)." )
    ap.add_argument("--key-label", default="KEY", help="Label string used to identify the key row in a non-item column (case-insensitive)." )

    args = ap.parse_args()
    input_csv = args.input
    if args.output:
        output_csv = args.output
    else:
        if input_csv.lower().endswith(".csv"):
            output_csv = re.sub(r"\.csv$", "", input_csv, flags=re.IGNORECASE) + "_with_item_stats.csv"
        else:
            output_csv = input_csv + "_with_item_stats.csv"

    run(
        input_csv=input_csv,
        output_csv=output_csv,
        item_pattern=args.item_pattern,
        percent=args.percent,
        label_col=args.label_col,
        exam_stats_csv=args.exam_stats_csv,
        plots_dir=args.plots_dir,
        key_row_index=args.key_row_index,
        answers_mode=args.answers_mode,
        item_report_csv=args.item_report_csv,
        key_label=args.key_label,
    )


if __name__ == "__main__":
    main()
