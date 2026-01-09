# PM Adverse Event Detection Tool

This repo has Python script for detecting **adverse particulate matter (PM) events** from time-resolved air quality data.

The tool is designed for environmental monitoring, exposure analysis, and research workflows
where identifying sustained, elevated PM concentrations relative to a daily baseline is required.

---

## Features

- Timezone-aware timestamp handling (UTC or local)
- Daily baseline calculation
- Percent-above-baseline adverse threshold
- Gap tolerance for noisy data
- Minimum event duration filtering
- Daily summary statistics
- Event-level output table
- Single combined plot for all data
- Interactive plot window with zoom and pan support

---

## Time Handling

You must specify:
- The timestamp column name
- Whether timestamps are UTC or already local
- The local timezone for analysis (IANA format)

Examples of valid timezones:
- `America/Denver`
- `America/Los_Angeles`
- `UTC`

If timestamps are UTC, they are converted to local time before analysis.  
All calculations are performed on **local calendar days** (midnight to midnight).

Optionally, you may explicitly specify the timestamp format using `--time-format`
(e.g. `%Y-%m-%d %H:%M:%S`).

---

## Daily Baseline Definition

For each local day (midnight to midnight), the baseline is defined as:

```
baseline = max(
    mean(PM from 00:00–03:00 local),
    baseline_floor
)
```

- Default `baseline_floor = 10`
- If no valid data exists between 00:00 and 03:00, or PM levels are low, the default baseline floor is used

---

## Adverse Threshold

The adverse threshold is defined as a percentage above the baseline:

```
threshold = baseline × (1 + threshold_pct / 100)
```

Example:
- `threshold_pct = 30`
- Threshold = 130% of baseline

---

## Adverse Event Definition

An adverse event is defined as:

- PM values above the daily threshold
- Sustained for at least `--min-duration-min` minutes
- After applying optional gap tolerance

Events shorter than the minimum duration are ignored.

---

## Gap Tolerance (Optional)

We can provide a flag to ignore short drops in PM. This is particularly important for monitors that are not performing well

If `--gap-tolerance-min` is specified, short dips below the threshold are filled
(merged) when:

- The dip duration is less than or equal to `gap-tolerance-min`
- The dip is surrounded on both sides by above-threshold values

This prevents artificial splitting of a single event into multiple events.

---

## Outputs

### Daily Summary Table (CSV optional)

Columns:
- `day`
- `day_start`
- `baseline`
- `threshold_value`
- `n_events`
- `median_duration_min`
- `iqr_duration_min`

### Event-Level Table (CSV optional)

Columns:
- `day`
- `day_start`
- `baseline`
- `threshold_value`
- `start`
- `end`
- `duration_min`
- `peak_pm`
- `mean_pm`

---

## Plotting

The script can generate a plot of the PM levels

- PM time series
- Points above the adverse threshold highlighted
- Adverse events shaded
- Daily baseline and threshold drawn
- Optional interactive window for zoom and pan

---

## Command-Line Arguments

This is how you control what the script is doing

### Required

- `--file`
- `--time-col`
- `--pm-col`
- `--time-scale`
- `--timezone`
- `--min-duration-min`
- `--threshold-pct`

### Optional

- `--time-format`
- `--gap-tolerance-min`
- `--baseline-floor`
- `--baseline-start-hour`
- `--baseline-end-hour`

### Output Control

- `--out-summary`
- `--out-events`
- `--print`

### Plotting

- `--plot`
- `--plot-path`
- `--plot-dpi`
- `--show-plot`
- `--no-save-plot`

---

## Example Commands

UTC timestamps converted to local time with interactive plot:

```
python event_detection.py --file pm_data.csv --time-col timestamp --pm-col pm25 --time-scale utc --timezone America/Denver --min-duration-min 20 --gap-tolerance-min 5 --threshold-pct 30 --plot --show-plot
```

Local timestamps, save outputs, no interactive window:

```
python event_detection.py --file pm_local.csv --time-col local_time --pm-col pm25 --time-scale local --timezone America/Los_Angeles --min-duration-min 30 --threshold-pct 50 --out-summary daily_summary.csv --out-events events.csv --plot
```

---

## Notes

- Interactive plotting requires a GUI-capable Python environment.
- For headless systems, omit `--show-plot`.

---