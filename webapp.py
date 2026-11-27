"""
webapp.py — interactive web front-end for the scouting-report pipeline.

A tiny FastAPI app: a styled form takes a player name, a pitcher/hitter choice,
and a date range, then renders the same HTML report the CLI produces — live.

Run locally:
    uvicorn webapp:app --reload

Deploy (Render): start command `uvicorn webapp:app --host 0.0.0.0 --port $PORT`.
"""

from __future__ import annotations

import base64
import html
import io
import sys
import traceback
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

# src/ is a sibling package dir; add it so `import pipeline` works when this
# module is run directly (uvicorn webapp:app) rather than as part of a package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import config
import pipeline
from pipeline import ReportError
from src import vaa_charts, vaa_season

# One season of four-seam results, built by build_vaa_table.py and committed.
_VAA_SEASON = 2026
_VAA_TABLE = config.PROJECT_ROOT / "app_data" / f"vaa_fastball_{_VAA_SEASON}.csv"
_VAA_HEIGHTS = (1.5, 4.0)        # comparison heights we accept, in feet

# Version comes from config.py (single source of truth), so the page footer and
# the FastAPI /docs metadata always agree with the report footer.
_VERSION = config.APP_VERSION

app = FastAPI(title="Automated Baseball Scouting Reports", version=_VERSION)

# Largest span we'll pull in one request. Keeps a live request from asking
# Statcast for something enormous (and slow). One year covers a full season.
_MAX_RANGE_DAYS = 366

_REPO_URL = "https://github.com/jmaxrod19/V1-agentic-baseball-scouting"

# Shared design tokens — same palette as the reports so the whole thing feels
# like one product.
_PAGE_CSS = """
  :root {
    --bg:#f5f1e8; --panel:#fffdf8; --ink:#1f2a34; --muted:#6b7280;
    --accent:#8b1e3f; --line:#d7d2c8;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--ink);
    font-family:Georgia,"Times New Roman",serif; line-height:1.5;
  }
  .wrap { max-width:640px; margin:0 auto; padding:48px 24px 64px; }
  .eyebrow {
    font-size:12px; text-transform:uppercase; letter-spacing:0.12em;
    color:var(--muted); margin-bottom:8px;
  }
  h1 { font-size:32px; line-height:1.1; margin:0 0 10px; }
  .lede { color:var(--muted); font-size:17px; margin:0 0 22px; }
  hr { border:none; border-top:3px solid var(--accent); margin:0 0 26px; }
  form { background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:22px; }
  label { display:block; font-weight:700; margin:14px 0 6px; }
  label:first-child { margin-top:0; }
  input[type=text], input[type=date] {
    width:100%; padding:10px 12px; font-size:16px; font-family:inherit;
    border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink);
  }
  .row { display:flex; gap:14px; }
  .row > div { flex:1; }
  .radios { display:flex; gap:18px; margin-top:4px; }
  .radios label { display:flex; align-items:center; gap:7px; font-weight:400; margin:0; }
  .btn {
    margin-top:20px; width:100%; padding:13px; font-size:17px; font-family:inherit;
    background:var(--accent); color:#fff; border:none; border-radius:8px; cursor:pointer;
  }
  .note { margin-top:14px; color:var(--muted); font-size:13px; }
  a { color:var(--accent); }
  .wide { max-width:1100px; padding-top:24px; }
  .chart { display:block; width:100%; height:auto; background:#fff;
           border:1px solid var(--line); border-radius:10px; }
  input[type=number] {
    width:100%; padding:10px 12px; font-size:16px; font-family:inherit;
    border:1px solid var(--line); border-radius:6px; background:#fff; color:var(--ink);
  }
  .picks li { margin:0 0 8px; }
  .error { background:var(--panel); border:1px solid var(--accent);
           border-radius:10px; padding:22px; }
  .error h2 { margin:0 0 8px; color:var(--accent); }
"""


def _page(body: str, title: str = "Scouting Reports", wide: bool = False) -> str:
    """Wrap page-body markup in the shared HTML shell. `wide` suits big charts."""
    return (
        "<!DOCTYPE html><html lang=en><head><meta charset=UTF-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1.0'>"
        f"<title>{title}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><div class='{'wrap wide' if wide else 'wrap'}'>{body}</div></body></html>"
    )


def _default_start() -> str:
    # Default to the full season so far (from opening day, ~April 1), not just the
    # current month — a month-to-date range is too thin for the bat-tracking
    # visuals (contact clusters, per-pitch zones) to have enough swings. In the
    # off-season (Jan–Mar) fall back to the previous completed season.
    today = date.today()
    year = today.year if today.month >= 4 else today.year - 1
    return date(year, 4, 1).isoformat()


def _default_end() -> str:
    today = date.today()
    if today.month >= 4:
        return today.isoformat()
    # Off-season: pair the previous-season start with that season's end.
    return date(today.year - 1, 10, 31).isoformat()


def _form_body(start: str, end: str) -> str:
    """The form markup, with the given default dates prefilled."""
    return f"""
      <div class=eyebrow>Automated Scouting Report</div>
      <h1>Build a scouting report</h1>
      <p class=lede>Enter a player, pick a date range, and get a live report —
         pitch arsenal for pitchers, batted-ball quality for hitters.</p>
      <hr>
      <p class=lede><a href=/vaa>New: see any {_VAA_SEASON} pitcher’s fastball approach angle →</a></p>
      <form method=post action=/report>
        <label for=name>Player name</label>
        <input id=name name=name type=text placeholder="e.g. Chase Burns" required>

        <label>Player type</label>
        <div class=radios>
          <label><input type=radio name=kind value=pitcher checked> Pitcher</label>
          <label><input type=radio name=kind value=hitter> Hitter</label>
        </div>

        <div class=row>
          <div>
            <label for=start>Start date</label>
            <input id=start name=start type=date value="{start}" required>
          </div>
          <div>
            <label for=end>End date</label>
            <input id=end name=end type=date value="{end}" required>
          </div>
        </div>

        <button class=btn type=submit>Generate report</button>
        <p class=note>Live data comes from MLB Statcast; the first request for a
           player may take a few seconds. Source on
           <a href="{_REPO_URL}">GitHub</a>. · v{_VERSION}</p>
      </form>
    """


def _error_body(message: str) -> str:
    # Escape: error messages embed the user's raw player-name input (e.g. "No
    # player found for '<name>'"), so rendering them unescaped would reflect
    # arbitrary HTML/script back into the page.
    return (
        f'<div class=error><h2>Can’t build that report</h2>'
        f'<p>{html.escape(message)}</p><p><a href="/">← Try again</a></p></div>'
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _page(_form_body(_default_start(), _default_end()))


@app.post("/report", response_class=HTMLResponse)
def report(
    name: str = Form(...),
    kind: str = Form(...),
    start: str = Form(...),
    end: str = Form(...),
) -> HTMLResponse:
    # -- Validate the date range before doing any expensive work -------------
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        e = datetime.strptime(end, "%Y-%m-%d").date()
    except ValueError:
        return HTMLResponse(_page(_error_body("Please use valid YYYY-MM-DD dates.")))

    if s > e:
        return HTMLResponse(_page(_error_body("Start date must be on or before the end date.")))
    if e - s > timedelta(days=_MAX_RANGE_DAYS):
        return HTMLResponse(_page(_error_body(
            "Please choose a range of a year or less."
        )))
    if e > date.today():
        return HTMLResponse(_page(_error_body(
            "Please choose a date range that has already happened."
        )))

    # -- Generate; expected failures come back as a styled error page --------
    try:
        result = pipeline.generate_report(name, start, end, kind)
    except ReportError as exc:
        return HTMLResponse(_page(_error_body(str(exc))))
    except Exception:  # noqa: BLE001 — last-resort guard so the app never 500s ugly
        # Log the real cause to stderr (Render captures it) so a genuine bug is
        # diagnosable, while the user still sees a friendly message.
        traceback.print_exc()
        return HTMLResponse(_page(_error_body(
            "Something went wrong pulling that data. Please try a different "
            "player or a shorter range."
        )))

    # result.html is a complete, self-contained report document.
    return HTMLResponse(result.html)


# ---------------------------------------------------------------------------
# Four-seam fastball approach angle (VAA): full season, any pitcher
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _vaa_table() -> pd.DataFrame:
    """The season table, read from disk once per process."""
    return pd.read_csv(_VAA_TABLE)


@lru_cache(maxsize=64)
def _vaa_png(pitcher_id: int, plate_height: float) -> tuple[str, str]:
    """(pitcher name, chart as base64 PNG). Cached: the season never changes."""
    c, height_in, league_height_in = vaa_season.comparison_from_table(
        _vaa_table(), pitcher_id, plate_height
    )
    buffer = io.BytesIO()
    # The report charts restyle matplotlib globally (serif fonts), so draw this
    # one with the default style. The lock keeps that switch from leaking.
    with pipeline.PLOT_LOCK, plt.style.context("default"):
        fig = vaa_charts.fastball_vaa_chart(c, height_in, league_height_in)
        try:
            fig.savefig(buffer, format="png", dpi=150, facecolor="white")
        finally:
            plt.close(fig)               # otherwise every request leaks a figure
    return c.name, base64.b64encode(buffer.getvalue()).decode("ascii")


def _vaa_form(value: str = "", height: float = 2.75) -> str:
    """Search box with name suggestions (a plain HTML datalist, no JavaScript)."""
    table = _vaa_table()
    pitchers = table.loc[table["role"] == "pitcher", "name"]
    names = sorted(pitchers.unique())            # two pitchers can share a name
    options = "".join(f'<option value="{html.escape(n, quote=True)}">' for n in names)
    return f"""
      <form method=get action=/vaa>
        <label for=name>Pitcher</label>
        <input id=name name=name type=text list=pitchers value="{html.escape(value, quote=True)}"
               placeholder="e.g. Tarik Skubal" autocomplete=off required>
        <datalist id=pitchers>{options}</datalist>
        <label for=height>Compare pitches crossing the plate at (feet)</label>
        <input id=height name=height type=number step=0.05 min={_VAA_HEIGHTS[0]}
               max={_VAA_HEIGHTS[1]} value="{height:g}">
        <button class=btn type=submit>Show approach angle</button>
        <p class=note>{len(pitchers)} pitchers who threw a four-seam fastball in the
           {_VAA_SEASON} regular season. <a href=/>← Scouting reports</a></p>
      </form>
    """


def _vaa_intro() -> str:
    return f"""
      <div class=eyebrow>{_VAA_SEASON} four-seam fastball</div>
      <h1>Vertical approach angle</h1>
      <p class=lede>How steeply does a pitcher’s fastball arrive at the plate
         compared with the league, and why?</p>
      <hr>
    """


def _vaa_message(message: str, value: str = "") -> HTMLResponse:
    """The search page again, with an explanation above the form."""
    note = f"<div class=error><p>{html.escape(message)}</p></div><br>"
    return HTMLResponse(_page(_vaa_intro() + note + _vaa_form(value), "Fastball VAA"))


@app.get("/vaa", response_class=HTMLResponse)
def vaa_page(name: str = "", pitcher: str = "", height: str = "2.75") -> HTMLResponse:
    """No arguments: search form. `name`: search. `pitcher` (MLBAM id): the chart."""
    if not _VAA_TABLE.exists():
        return HTMLResponse(_page(_error_body("The VAA season data is not installed.")))

    try:
        plate_height = round(float(height), 2)
    except ValueError:
        plate_height = float("nan")
    # NaN fails every comparison, so this one test also rejects non-numbers.
    if not _VAA_HEIGHTS[0] <= plate_height <= _VAA_HEIGHTS[1]:
        return _vaa_message(
            f"Pick a plate height between {_VAA_HEIGHTS[0]:g} and {_VAA_HEIGHTS[1]:g} feet.", name
        )

    if pitcher:
        if not pitcher.isdecimal() or len(pitcher) > 9:
            return _vaa_message("That pitcher link is not valid. Search by name instead.")
        pitcher_id = int(pitcher)
    elif name.strip():
        matches = vaa_season.find_pitchers(_vaa_table(), name)
        if matches.empty:
            return _vaa_message(
                f"No four-seam fastball found for '{name}' in {_VAA_SEASON}. Check the "
                "spelling; pitchers who never threw one are not listed.", name
            )
        if len(matches) > 1:
            items = "".join(
                f"<li><a href='/vaa?pitcher={int(m.pitcher_id)}&amp;height={plate_height:g}'>"
                f"{html.escape(m.name)}</a> ({m.hand}HP, {int(m.pitches):,} four-seamers)</li>"
                for m in matches.head(30).itertuples()
            )
            more = "" if len(matches) <= 30 else "<p class=note>Showing 30. Type more of the name.</p>"
            body = f"{_vaa_intro()}<p>Which pitcher?</p><ul class=picks>{items}</ul>{more}"
            return HTMLResponse(_page(body + _vaa_form(name, plate_height), "Fastball VAA"))
        pitcher_id = int(matches.iloc[0]["pitcher_id"])
    else:
        return HTMLResponse(_page(_vaa_intro() + _vaa_form(height=plate_height), "Fastball VAA"))

    try:
        shown_name, png = _vaa_png(pitcher_id, plate_height)
    except ValueError as problem:
        return _vaa_message(str(problem))
    except Exception:  # noqa: BLE001 — same last-resort guard as /report
        traceback.print_exc()
        return _vaa_message("Something went wrong drawing that chart.")

    safe_name = html.escape(shown_name, quote=True)
    body = f"""
      <p><a href=/vaa>← Search another pitcher</a></p>
      <img class=chart alt="{safe_name}: {_VAA_SEASON} four-seam fastball approach angle compared
           with the MLB average" src="data:image/png;base64,{png}">
      <br>
      {_vaa_form(shown_name, plate_height)}
    """
    return HTMLResponse(_page(body, f"{safe_name} · Fastball VAA", wide=True))
