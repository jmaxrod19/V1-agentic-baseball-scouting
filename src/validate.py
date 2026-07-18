"""Data validation for pulled Statcast frames.

A freshly pulled (or cache-reloaded) DataFrame is *inspected* here before the
pipeline trusts it. This module never raises on its own — it returns a
ValidationReport describing what it found, and the caller decides what to do
(the pipeline logs warnings and raises only on structural errors).

Two severities:
    error   — the frame is structurally broken (no rows, missing a column the
              metrics need). A report built from it would be garbage.
    warning — the frame is usable but something looks off (implausible values,
              an all-null column, duplicate rows). Worth surfacing, not fatal.

The plausibility bounds below are deliberately generous *sanity* limits (is this
physically possible?), not exact baseball definitions — a value outside them
signals bad data, not an unusual player.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Columns the downstream metrics genuinely require; absent => error.
_REQUIRED_COLS = ("game_date", "pitch_type")

# column -> (low, high) physically-plausible inclusive range.
_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "release_speed": (30.0, 106.0),   # mph — slowest eephus to hardest fastball
    "launch_speed": (0.0, 125.0),     # mph — exit velocity
    "launch_angle": (-90.0, 90.0),    # degrees — straight down to straight up
    "release_spin_rate": (0.0, 4000.0),  # rpm — elite breaking balls reach ~3700
}


@dataclass
class ValidationIssue:
    """One thing the validator noticed, with its severity."""
    severity: str   # "error" | "warning"
    message: str


@dataclass
class ValidationReport:
    """The result of validating a frame: row count + a list of issues."""
    n_rows: int
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, message))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when there are no error-level issues (warnings are allowed)."""
        return not self.errors

    def summary(self) -> str:
        """One-line human summary, e.g. for logs or an exception message."""
        if not self.issues:
            return f"{self.n_rows} rows — no issues"
        parts = "; ".join(f"[{i.severity}] {i.message}" for i in self.issues)
        return (
            f"{self.n_rows} rows — "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s): {parts}"
        )


def validate_statcast(df: pd.DataFrame) -> ValidationReport:
    """Inspect a Statcast-schema DataFrame and return a ValidationReport."""
    report = ValidationReport(n_rows=len(df))

    # 1. Any rows at all? An empty frame can't produce a report.
    if len(df) == 0:
        report.add("error", "frame has no rows")
        return report  # nothing else is meaningful on an empty frame

    # 2. Required columns present? Their absence breaks the metrics.
    for col in _REQUIRED_COLS:
        if col not in df.columns:
            report.add("error", f"missing required column '{col}'")

    # 3. Was the date column actually parsed to datetimes? If not, the loader was
    #    likely bypassed — a warning, since date math downstream would misbehave.
    if "game_date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["game_date"]
    ):
        report.add("warning", "game_date is not datetime dtype")

    # 4. Plausibility: count values outside the physical range for each column
    #    that's present. An all-null column is its own (softer) warning.
    for col, (lo, hi) in _PLAUSIBLE.items():
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if values.empty:
            report.add("warning", f"'{col}' is entirely null")
            continue
        n_bad = int(((values < lo) | (values > hi)).sum())
        if n_bad:
            report.add(
                "warning",
                f"{n_bad} '{col}' value(s) outside plausible range [{lo}, {hi}]",
            )

    # 5. Fully-duplicate rows usually mean a merge/pull glitch, not real pitches.
    n_dup = int(df.duplicated().sum())
    if n_dup:
        report.add("warning", f"{n_dup} fully-duplicate row(s)")

    return report
