# =============================================================================
# Project: Adverse PM Event Detection
# Script: event_detection.py
# Purpose: Identify periods of elevated PM as a trigger for intervention.
# Author: Dr. Christian L’Orange | Associate Professor, CSU Mechanical Eng.
# Co-author: Carsen Hobson
# Contact: christian.lorange@colostate.edu | GitHub: https://github.com/orgs/LOrangeResearch
# Created: 2025-08-19
# Last Updated: 2026-01-08
# Version: 1.1.9
# =============================================================================


# import required packages
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    import pytz

    class ZoneInfo:  # type: ignore
        def __new__(cls, name: str):
            return pytz.timezone(name)


@dataclass
class Results:
    daily_summary: pd.DataFrame
    events: pd.DataFrame

# helper functions

def _parse_time_column(
    s: pd.Series,
    time_scale: str,
    tz_name: str,
    time_format: Optional[str] = None,
) -> pd.DatetimeIndex:
    if time_format:
        t = pd.to_datetime(s, errors="coerce", format=time_format, utc=False)
    else:
        t = pd.to_datetime(s, errors="coerce", utc=False)

    if t.isna().any():
        bad = s[t.isna()].head(8).tolist()
        raise ValueError(
            f"Failed to parse {t.isna().sum()} timestamps. Examples: {bad}. "
            f"Consider supplying --time-format."
        )

    local_tz = ZoneInfo(tz_name)

    if getattr(t.dt, "tz", None) is not None:
        if time_scale.lower() == "utc":
            t = t.dt.tz_convert(ZoneInfo("UTC")).dt.tz_convert(local_tz)
        elif time_scale.lower() == "local":
            t = t.dt.tz_convert(local_tz)
        else:
            raise ValueError("time_scale must be 'utc' or 'local'")
    else:
        if time_scale.lower() == "utc":
            t = t.dt.tz_localize(ZoneInfo("UTC")).dt.tz_convert(local_tz)
        elif time_scale.lower() == "local":
            t = t.dt.tz_localize(local_tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            raise ValueError("time_scale must be 'utc' or 'local'")

    return pd.DatetimeIndex(t)


def _iqr(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def _estimate_typical_dt_seconds(times: pd.Series) -> float:
    diffs = times.diff().dt.total_seconds().dropna().values
    if diffs.size == 0:
        return 0.0
    return max(float(np.median(diffs)), 0.0)


def _fill_short_false_gaps(times: pd.Series, above: np.ndarray, gap_tolerance_min: float) -> np.ndarray:
    """
    we fill short times where pm drops
    """
    if gap_tolerance_min <= 0 or above.size == 0 or not above.any():
        return above

    tvals = times.to_numpy()  # pandas.Timestamp array
    filled = above.copy()
    idx = np.arange(len(above))

    false_starts = idx[(~above) & (np.r_[False, above[:-1]])]
    false_ends = idx[(~above) & (np.r_[above[1:], False])]

    for fs, fe in zip(false_starts, false_ends):
        if fs == 0 or fe == len(above) - 1:
            continue
        if not (above[fs - 1] and above[fe + 1]):
            continue

        gap_seconds = (tvals[fe] - tvals[fs]).total_seconds()
        if (gap_seconds / 60.0) <= gap_tolerance_min:
            filled[fs : fe + 1] = True

    return filled


def analyze_pm_events(
    df: pd.DataFrame,
    time_col: str,
    pm_col: str,
    time_scale: str,
    timezone_name: str,
    min_duration_min: float,
    threshold_pct: float,
    gap_tolerance_min: float = 0.0,
    time_format: Optional[str] = None,
    baseline_floor: float = 10.0,
    baseline_start_hour: int = 0,
    baseline_end_hour: int = 3,
) -> Results:
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in columns: {list(df.columns)}")
    if pm_col not in df.columns:
        raise KeyError(f"pm_col '{pm_col}' not found in columns: {list(df.columns)}")

    out = df[[time_col, pm_col]].copy()
    out["_t"] = _parse_time_column(out[time_col], time_scale, timezone_name, time_format=time_format)
    out["_pm"] = pd.to_numeric(out[pm_col], errors="coerce")
    out = out.dropna(subset=["_t", "_pm"]).sort_values("_t").reset_index(drop=True)

    if out.empty:
        daily = pd.DataFrame(
            columns=["day", "day_start", "baseline", "threshold_value", "n_events", "median_duration_min", "iqr_duration_min"]
        )
        events = pd.DataFrame(
            columns=["day", "day_start", "baseline", "threshold_value", "start", "end", "duration_min", "peak_pm", "mean_pm"]
        )
        return Results(daily_summary=daily, events=events)

    out["_day"] = out["_t"].dt.floor("D")

    daily_rows = []
    event_rows = []
    mult = 1.0 + float(threshold_pct) / 100.0

    for day_start, g in out.groupby("_day", sort=True):
        g = g.sort_values("_t").reset_index(drop=True)
        typical_dt_s = _estimate_typical_dt_seconds(g["_t"])

        b0 = day_start + pd.Timedelta(hours=baseline_start_hour)
        b1 = day_start + pd.Timedelta(hours=baseline_end_hour)
        bw = g[(g["_t"] >= b0) & (g["_t"] < b1)]

        baseline_mean = float(bw["_pm"].mean()) if not bw.empty else float("nan")
        baseline = float(baseline_floor if np.isnan(baseline_mean) else max(baseline_mean, baseline_floor))
        threshold_value = baseline * mult

        above = (g["_pm"].to_numpy() > threshold_value)
        above = _fill_short_false_gaps(g["_t"], above, gap_tolerance_min)

        idx = np.arange(len(above))
        starts = idx[(above) & (~np.r_[False, above[:-1]])]
        ends = idx[(above) & (~np.r_[above[1:], False])]  # inclusive

        durations = []

        for s_i, e_i in zip(starts, ends):
            start_t = g.loc[s_i, "_t"]
            last_t = g.loc[e_i, "_t"]
            end_t = last_t + pd.Timedelta(seconds=typical_dt_s)

            duration_min = (end_t - start_t).total_seconds() / 60.0
            if duration_min < float(min_duration_min):
                continue

            seg = g.loc[s_i:e_i, "_pm"]
            event_rows.append(
                dict(
                    day=day_start.date().isoformat(),
                    day_start=day_start,
                    baseline=baseline,
                    threshold_value=threshold_value,
                    start=start_t,
                    end=end_t,
                    duration_min=float(duration_min),
                    peak_pm=float(seg.max()),
                    mean_pm=float(seg.mean()),
                )
            )
            durations.append(float(duration_min))

        durations = np.array(durations, dtype=float)

        daily_rows.append(
            dict(
                day=day_start.date().isoformat(),
                day_start=day_start,
                baseline=baseline,
                threshold_value=threshold_value,
                n_events=int(durations.size),
                median_duration_min=float(np.median(durations)) if durations.size else np.nan,
                iqr_duration_min=_iqr(durations) if durations.size else np.nan,
            )
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("day_start").reset_index(drop=True)
    events_df = pd.DataFrame(event_rows)
    if not events_df.empty:
        events_df = events_df.sort_values(["day_start", "start"]).reset_index(drop=True)

    return Results(daily_summary=daily_df, events=events_df)


def plot_all_events(
    clean: pd.DataFrame,
    daily_summary: pd.DataFrame,
    events: pd.DataFrame,
    plot_path: Optional[str],
    dpi: int,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    if clean.empty:
        return

    daily = daily_summary.set_index("day_start")

    clean = clean.copy()
    clean["baseline"] = clean["_day"].map(daily["baseline"])
    clean["threshold"] = clean["_day"].map(daily["threshold_value"])
    clean["above_threshold"] = clean["_pm"] > clean["threshold"]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Base time series
    ax.plot(clean["_t"], clean["_pm"], linewidth=1, label="PM")

    # Color points above threshold
    above = clean[clean["above_threshold"]]
    if not above.empty:
        ax.scatter(above["_t"], above["_pm"], s=12, label="PM (above threshold)", zorder=3)

    # Step lines per day
    for day_start, row in daily.iterrows():
        day_end = day_start + pd.Timedelta(days=1)
        ax.hlines(row["baseline"], xmin=day_start, xmax=day_end, linestyles="--", linewidth=1)
        ax.hlines(row["threshold_value"], xmin=day_start, xmax=day_end, linestyles=":", linewidth=1)

    # Legend handles for baseline/threshold once
    ax.plot([], [], linestyle="--", linewidth=1, label="Baseline")
    ax.plot([], [], linestyle=":", linewidth=1, label="Threshold")

    # Shade events
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            ax.axvspan(ev["start"], ev["end"], alpha=0.25)

    ax.set_title("PM Time Series with Adverse Events (All Data)")
    ax.set_xlabel("Time (local)")
    ax.set_ylabel("PM")
    ax.legend(loc="best")
    fig.tight_layout()

    if plot_path:
        fig.savefig(plot_path, dpi=dpi)

    if show:
        print("Opening interactive plot window (close the window to let the script exit)...")
        plt.show(block=True)  # <-- IMPORTANT
        # Do NOT close the figure before show
    else:
        plt.close(fig)


def load_table(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    if sep is None:
        return pd.read_csv(path, engine="python")
    return pd.read_csv(path, sep=sep)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect adverse PM events relative to daily baseline.")
    p.add_argument("--file", required=True, help="Path to input table (CSV/TSV).")
    p.add_argument("--sep", default=None, help="Delimiter (e.g., ',' or '\\t'). If omitted, tries to infer.")
    p.add_argument("--time-col", required=True, help="Name of timestamp column.")
    p.add_argument("--pm-col", required=True, help="Name of PM column.")
    p.add_argument("--time-format", default=None, help='Optional strptime format, e.g. "%Y-%m-%d %H:%M:%S".')
    p.add_argument("--time-scale", choices=["utc", "local"], required=True)
    p.add_argument("--timezone", required=True)

    p.add_argument("--min-duration-min", type=float, required=True)
    p.add_argument("--gap-tolerance-min", type=float, default=0.0)
    p.add_argument("--threshold-pct", type=float, required=True)

    p.add_argument("--baseline-floor", type=float, default=10.0)
    p.add_argument("--baseline-start-hour", type=int, default=0)
    p.add_argument("--baseline-end-hour", type=int, default=3)

    p.add_argument("--out-summary", default=None)
    p.add_argument("--out-events", default=None)
    p.add_argument("--print", action="store_true")

    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="pm_events_all.png")
    p.add_argument("--plot-dpi", type=int, default=150)
    p.add_argument("--show-plot", action="store_true")
    p.add_argument("--no-save-plot", action="store_true")

    return p

# main functions
def main() -> int:
    args = build_argparser().parse_args()
    df = load_table(args.file, sep=args.sep)

    res = analyze_pm_events(
        df=df,
        time_col=args.time_col,
        pm_col=args.pm_col,
        time_scale=args.time_scale,
        timezone_name=args.timezone,
        min_duration_min=args.min_duration_min,
        threshold_pct=args.threshold_pct,
        gap_tolerance_min=args.gap_tolerance_min,
        time_format=args.time_format,
        baseline_floor=args.baseline_floor,
        baseline_start_hour=args.baseline_start_hour,
        baseline_end_hour=args.baseline_end_hour,
    )

    if args.out_summary:
        res.daily_summary.to_csv(args.out_summary, index=False)
    if args.out_events:
        res.events.to_csv(args.out_events, index=False)

    if args.print or (not args.out_summary and not args.out_events):
        print(res.daily_summary.to_csv(index=False).strip())

    if args.plot:
        clean = df[[args.time_col, args.pm_col]].copy()
        clean["_t"] = _parse_time_column(
            clean[args.time_col],
            time_scale=args.time_scale,
            tz_name=args.timezone,
            time_format=args.time_format,
        )
        clean["_pm"] = pd.to_numeric(clean[args.pm_col], errors="coerce")
        clean = clean.dropna(subset=["_t", "_pm"]).sort_values("_t").reset_index(drop=True)
        if not clean.empty:
            clean["_day"] = clean["_t"].dt.floor("D")

            plot_path = None if args.no_save_plot else args.plot_path
            plot_all_events(
                clean=clean,
                daily_summary=res.daily_summary,
                events=res.events,
                plot_path=plot_path,
                dpi=args.plot_dpi,
                show=args.show_plot,
            )
        else:
            print("No valid (time, pm) rows to plot after parsing/coercion.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
