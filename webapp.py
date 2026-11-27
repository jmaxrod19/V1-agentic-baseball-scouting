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
from src.vaa_model import ZONE_PARTS, not_primary_words

# One season of four-seam results, built by build_vaa_table.py and committed.
_VAA_SEASON = 2026
_VAA_TABLE = config.PROJECT_ROOT / "app_data" / f"vaa_fastball_{_VAA_SEASON}.csv"

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
  .row > div { flex:1; min-width:0; }
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
  .zonepanel { display:flex; flex-wrap:wrap; gap:22px; align-items:flex-start;
               background:var(--panel); border:1px solid var(--line);
               border-radius:10px; padding:20px; margin:0 0 18px; }
  .zonepick { width:170px; text-align:center; }
  .zone { border:3px solid var(--ink); }
  .zone a, .allzone { display:block; text-decoration:none; color:var(--ink); background:#fff; }
  .zone a { padding:13px 0; border-top:1px solid var(--line); }
  .zone a:first-child { border-top:none; }
  .allzone { margin-top:8px; padding:8px 0; border:1px solid var(--line); border-radius:6px; }
  .zone a.on, .allzone.on { background:var(--accent); color:#fff; font-weight:700; }
  .why { flex:1; min-width:240px; }
  .why h2 { font-size:19px; margin:0 0 8px; }
  .why p { margin:0 0 10px; }
  .story { background:var(--panel); border:1px solid var(--line); border-radius:10px;
           padding:8px 24px 18px; margin:18px 0; font-size:17px; }
  .story > * { max-width:68ch; }
  .story h2 { font-size:24px; line-height:1.2; margin:18px 0 10px; }
  .story h3 { font-size:18px; margin:24px 0 6px; color:var(--accent); }
  .story p { margin:0 0 10px; }
  .feature { display:block; background:var(--ink); color:#fff; text-decoration:none;
             border-radius:14px; padding:24px; margin:40px 0 0;
             box-shadow:0 8px 22px rgba(31,42,52,0.22); transition:transform .15s; }
  .feature:hover { transform:translateY(-3px); }
  .feature:focus-visible { outline:3px solid var(--accent); outline-offset:3px; }
  .feature svg { display:block; width:100%; height:auto; margin:0 0 14px; }
  .badge { display:inline-block; background:var(--accent); border-radius:20px;
           padding:3px 12px; font-size:12px; font-weight:700; letter-spacing:0.1em;
           text-transform:uppercase; }
  .feature h2 { font-size:28px; line-height:1.15; margin:12px 0 8px; }
  .feature p { margin:0 0 18px; color:#d9dde3; }
  .cta { display:inline-block; background:#fffdf8; color:var(--accent); font-weight:700;
         font-size:17px; border-radius:8px; padding:12px 20px; }
  .search { padding:14px 18px; margin:0 0 18px; }
  .searchrow { display:flex; flex-wrap:wrap; gap:10px; }
  .searchrow input { flex:100 1 220px; width:auto; }
  .searchrow .btn { flex:1 0 auto; width:auto; margin:0; padding:10px 18px; }
  .search .note { margin:8px 0 0; }
  .notprimary { background:var(--panel); border:1px solid var(--line);
                border-left:6px solid var(--accent); border-radius:8px;
                padding:12px 18px; margin:0 0 18px; }
  .wide .note { font-size:15px; }
  .method { background:var(--panel); border:1px solid var(--line); border-radius:10px;
            padding:12px 18px; margin:12px 0 0; }
  .method summary { cursor:pointer; font-weight:700; }
  .method li { margin:0 0 6px; }
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


# Home-page advert for the VAA page. The whole card is one link. The drawing is
# decoration (not data): two fastballs ending at the same spot, one flatter.
_VAA_FEATURE = f"""
  <a class=feature href=/vaa>
    <svg viewBox="0 0 600 120" aria-hidden=true fill=none stroke-linecap=round>
      <path d="M8 108 H592" stroke="#4b5766" stroke-width=2 />
      <rect x=546 y=46 width=22 height=50 stroke="#93c5fd" stroke-width=2 />
      <path d="M24 16 Q 320 22 557 72" stroke="#9ca3af" stroke-width=4 stroke-dasharray="2 10" />
      <path d="M24 56 Q 320 52 557 72" stroke="#ee5a77" stroke-width=5 />
      <circle cx=24 cy=56 r=9 fill="#fff" />
    </svg>
    <span class=badge>New · {_VAA_SEASON} season</span>
    <h2>How flat is that fastball?</h2>
    <p>Search any pitcher’s four-seamer. See its vertical approach angle next to the
       MLB average, and why it comes in that way.</p>
    <span class=cta>Explore Fastball VAA →</span>
  </a>
"""


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
      {_VAA_FEATURE}
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


@lru_cache(maxsize=96)
def _vaa_png(pitcher_id: int, part: str) -> str:
    """The chart as a base64 PNG. Cached: the season never changes."""
    c, height_in, league_height_in = vaa_season.comparison_from_table(
        _vaa_table(), pitcher_id, part
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
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _vaa_pitchers() -> pd.DataFrame:
    """One row per pitcher (the all-heights rows)."""
    table = _vaa_table()
    return table[(table["role"] == "pitcher") & (table["part"] == "all")]


def _vaa_form(value: str = "", part: str = "all") -> str:
    """One-line search bar with name suggestions (an HTML datalist, no JavaScript)."""
    pitchers = _vaa_pitchers()["name"]
    names = sorted(pitchers.unique())            # two pitchers can share a name
    options = "".join(f'<option value="{html.escape(n, quote=True)}">' for n in names)
    return f"""
      <form class=search method=get action=/vaa>
        <label for=name>Search a pitcher</label>
        <div class=searchrow>
          <input id=name name=name type=text list=pitchers value="{html.escape(value, quote=True)}"
                 placeholder="e.g. Tarik Skubal" autocomplete=off required>
          <button class=btn type=submit>Show approach angle</button>
        </div>
        <datalist id=pitchers>{options}</datalist>
        <input type=hidden name=zone value={part}>
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


# Plain explanation of why approach angle matters. It teaches the idea and quotes
# no results, so it stays true for any season.
_TAKEAWAYS = {
    "all": ("Why approach angle matters",
            "Every fastball is falling when it reaches the plate. How steeply it falls changes "
            "where a hitter has to put the bat, and that matters differently high and low. "
            "Pick a part of the zone."),
    "top": ("Top of the zone: where flat plays",
            "A flatter fastball drops less than a hitter expects, so it can seem to rise and "
            "stay above the barrel. This is where that shape tends to help most."),
    "middle": ("Middle of the zone: less room for error",
               "Belt high is where hitters are most comfortable. An unusual angle can still "
               "help, but shape alone offers less protection here."),
    "bottom": ("Bottom of the zone: a different job",
               "Down low, a flat fastball can lose its edge and run into the path of the bat. "
               "A steeper one keeps working downhill, which can help keep the ball on the "
               "ground."),
}


def _takeaway(part: str) -> str:
    """The short explanation beside the picker, for the chosen part of the zone."""
    title, text = _TAKEAWAYS[part]
    return f"<h2>{title}</h2><p>{text}</p>"


_VAA_STORY = """
  <article class=story id=story>
    <h2>Shape, location and the swing</h2>
    <p>Vertical approach angle (VAA) is how steeply a pitch is coming down as it reaches
       the plate. It is written as a negative number: the closer to zero, the flatter the
       pitch. A flat angle is not “good” and a steep one is not “bad.” It is a question
       of fit.</p>

    <h3>No fastball rises. Some just fall less.</h3>
    <p>A hitter has to decide where the ball will be before it gets there, going on all the
       fastballs they have seen. A flatter one drops less than usual, so it shows up above
       where the barrel is headed. That is the “ride” or “rise” people talk about, and it
       can lead to swings underneath the ball: misses and pop-ups.</p>

    <h3>Location changes what the angle does</h3>
    <p>At the top of the zone, a flat fastball plays to its strength. Lower in the zone, the
       same pitch can be easier to square up. A steeper fastball travels more downhill,
       which tends to suit the lower part of the zone and contact on the ground.</p>

    <h3>It works together with velocity and the swing</h3>
    <p>Velocity cuts the time a hitter has to adjust, so a hard fastball with an unusual
       angle asks two hard questions at once. Swing path matters too: most swings travel
       slightly upward at contact, and some match certain angles better than others. What
       fools one hitter may be right in another’s wheelhouse.</p>

    <h3>The pitcher’s trade-off</h3>
    <p>Going up can draw swings under the ball, but a miss too high is easy to take. Going
       down can get called strikes and ground balls, but it asks more of a flat fastball.
       The best spot depends on the fastball a pitcher has and who is hitting.</p>

    <h3>Why compare at the same height</h3>
    <p>Aim alone changes the angle: a high fastball arrives flatter than a low one, even from
       the same arm. A pitcher who lives upstairs could look “flat” just because of where they
       throw. Comparing fastballs that crossed at similar heights shows what belongs to the
       pitcher: release point, carry and velocity.</p>
  </article>
"""


def _zone_panel(pitcher_id: int, part: str) -> str:
    """Pick a third of the zone (drawn as a strike zone), with what it means."""
    def link(key: str, text: str, css: str = "") -> str:
        on = " on" if key == part else ""
        current = " aria-current=true" if key == part else ""
        return (f"<a class='{css}{on}'{current} "
                f"href='/vaa?pitcher={pitcher_id}&amp;zone={key}'>{text}</a>")

    thirds = link("top", "Top third") + link("middle", "Middle third") + link("bottom", "Bottom third")
    return f"""
      <div class=zonepanel>
        <div class=zonepick>
          <p class=eyebrow>Pick a part of the zone</p>
          <div class=zone>{thirds}</div>
          {link("all", "All heights", "allzone")}
        </div>
        <div class=why>
          {_takeaway(part)}
          <p><a href=#story>Read more ↓</a></p>
          <p class=note>Thirds are of each batter’s strike zone. The chart compares him
             only with fastballs from the same part.</p>
        </div>
      </div>
    """


def _method_panel(pitcher_id: int, part: str) -> str:
    """The chart's method notes, folded away until a reader asks for them."""
    c = vaa_season.comparison_from_table(_vaa_table(), pitcher_id, part)[0]
    items = "".join(f"<li>{html.escape(note)}</li>" for note in vaa_charts.method_notes(c))
    return f"<details class=method><summary>How this is calculated</summary><ul>{items}</ul></details>"


def _primary_note(pitcher: pd.Series) -> str:
    """A plain warning when the four-seamer was not this pitcher's main pitch."""
    flag = not_primary_words(pitcher.get("ff_pct"), pitcher.get("main_pitch"),
                             pitcher.get("main_pct"))
    if flag is None:
        return ""
    main, four_seam, most = flag
    return (f"<div class=notprimary><b>Not his primary pitch.</b> "
            f"{html.escape(str(pitcher['name']))} threw the four-seamer {four_seam} of the time "
            f"in {_VAA_SEASON}. His main pitch was the {html.escape(main)} ({most}).</div>")


def _vaa_message(message: str, value: str = "") -> HTMLResponse:
    """The search page again, with an explanation above the form."""
    note = f"<div class=error><p>{html.escape(message)}</p></div><br>"
    return HTMLResponse(_page(_vaa_intro() + note + _vaa_form(value), "Fastball VAA"))


@app.get("/vaa", response_class=HTMLResponse)
def vaa_page(name: str = "", pitcher: str = "", zone: str = "all") -> HTMLResponse:
    """No arguments: search form. `name`: search. `pitcher` (MLBAM id): the chart."""
    if not _VAA_TABLE.exists():
        return HTMLResponse(_page(_error_body("The VAA season data is not installed.")))
    if zone not in ZONE_PARTS:
        return _vaa_message("That part of the zone is not an option. Search again.", name)

    if pitcher:
        # MLBAM ids are six digits; refusing very long input avoids an int() size error.
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
                f"<li><a href='/vaa?pitcher={int(m.pitcher_id)}&amp;zone={zone}'>"
                f"{html.escape(m.name)}</a> ({m.hand}HP, {int(m.pitches):,} four-seamers)</li>"
                for m in matches.head(30).itertuples()
            )
            more = "" if len(matches) <= 30 else "<p class=note>Showing 30. Type more of the name.</p>"
            choices = f"<p>Which pitcher?</p><ul class=picks>{items}</ul>{more}"
            body = _vaa_intro() + _vaa_form(name, zone) + choices
            return HTMLResponse(_page(body, "Fastball VAA"))
        pitcher_id = int(matches.iloc[0]["pitcher_id"])
    else:
        return HTMLResponse(_page(_vaa_intro() + _vaa_form(part=zone), "Fastball VAA"))

    pitchers = _vaa_pitchers()
    mine = pitchers[pitchers["pitcher_id"] == pitcher_id]
    if mine.empty:
        return _vaa_message("No four-seam fastball data for that pitcher this season.")
    shown_name = str(mine["name"].iloc[0])
    safe_name = html.escape(shown_name, quote=True)

    try:
        png = _vaa_png(pitcher_id, zone)
        chart = (f'<img class=chart alt="{safe_name}: {_VAA_SEASON} four-seam fastball approach '
                 f'angle, {ZONE_PARTS[zone]}, compared with the MLB average" '
                 f'src="data:image/png;base64,{png}">') + _method_panel(pitcher_id, zone)
    except ValueError as problem:
        # e.g. never threw one in this third: keep the picker so another can be chosen.
        chart = f"<div class=error><p>{html.escape(str(problem))}</p></div>"
    except Exception:  # noqa: BLE001 — same last-resort guard as /report
        traceback.print_exc()
        return _vaa_message("Something went wrong drawing that chart.")

    body = f"""
      {_vaa_form(shown_name, zone)}
      {_primary_note(mine.iloc[0])}
      {_zone_panel(pitcher_id, zone)}
      {chart}
      {_VAA_STORY}
    """
    return HTMLResponse(_page(body, f"{safe_name} · Fastball VAA", wide=True))
