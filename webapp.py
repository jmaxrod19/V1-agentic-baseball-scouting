"""
webapp.py — interactive web front-end for the scouting-report pipeline.

A tiny FastAPI app: a styled form takes a player name, a pitcher/hitter choice,
and a date range, then renders the same HTML report the CLI produces — live.

Run locally:
    uvicorn webapp:app --reload

Deploy (Render): start command `uvicorn webapp:app --host 0.0.0.0 --port $PORT`.
"""

from __future__ import annotations

import html
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

# src/ is a sibling package dir; add it so `import pipeline` works when this
# module is run directly (uvicorn webapp:app) rather than as part of a package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

import config
import pipeline
from pipeline import ReportError

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
  .error { background:var(--panel); border:1px solid var(--accent);
           border-radius:10px; padding:22px; }
  .error h2 { margin:0 0 8px; color:var(--accent); }
"""


def _page(body: str, title: str = "Scouting Reports") -> str:
    """Wrap page-body markup in the shared HTML shell."""
    return (
        "<!DOCTYPE html><html lang=en><head><meta charset=UTF-8>"
        "<meta name=viewport content='width=device-width, initial-scale=1.0'>"
        f"<title>{title}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><div class=wrap>{body}</div></body></html>"
    )


def _default_start() -> str:
    return date.today().replace(day=1).isoformat()


def _default_end() -> str:
    return date.today().isoformat()


def _form_body(start: str, end: str) -> str:
    """The form markup, with the given default dates prefilled."""
    return f"""
      <div class=eyebrow>Automated Scouting Report</div>
      <h1>Build a scouting report</h1>
      <p class=lede>Enter a player, pick a date range, and get a live report —
         pitch arsenal for pitchers, batted-ball quality for hitters.</p>
      <hr>
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
