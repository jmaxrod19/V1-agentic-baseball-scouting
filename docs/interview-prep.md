# Talking about this project — interview notes

Structure and reasoning, not syntax. Read this the way you'd say it out loud —
adapt the wording to your own voice, but keep the shape and the "why" behind
each answer. Pairs with `docs/sql-notes.md` if a conversation goes deep on the
SQL specifically.

## The opener (memorize this one)

"I built a baseball scouting analytics tool that turns raw Statcast and
Baseball Savant data into professional-grade scouting reports — the kind a
front office would actually use. It started as a script, grew into a deployed
web app, and now includes an LLM-written narrative layer and a from-scratch
biomechanics visualization backed by SQL."

That's the whole pitch. Everything below is what you say on a follow-up.

## "What's it built with?"

Keep this short and plain-language — don't recite every library:

- **Python** for everything
- **pandas** for data wrangling
- **DuckDB** for the SQL analysis layer (a good story — see below)
- **matplotlib** for the visuals (deliberately static, not interactive)
- **FastAPI** for the web app
- **The Claude API** for the narrative-writing feature
- **GitHub Actions** for CI, auto-deployed to Hugging Face Spaces

## "Walk me through the structure"

Five layers, each with one job. You don't need to name files — just the shape:

1. **Data layer.** Pulls from four different sources — pybaseball, Baseball
   Savant's own CSV exports, FanGraphs, and MLB's stats API — each with its own
   quirks, cleaned into one consistent shape. Raw pulls are kept untouched;
   cleaned versions are cached separately, so the same input always produces
   the same output.

2. **Analysis layer.** Turns clean data into real stats — pitch arsenals,
   batted-ball quality, league percentile rankings. This is where the SQL work
   lives.

3. **Presentation layer.** Renders those stats into charts and a styled report.

4. **Orchestration layer.** Ties it together — takes a player name and a date
   range, resolves the name, pulls the data, runs the analysis, builds the
   report.

5. **Delivery.** A small web app on top, so anyone can type in a name and get
   a live report.

## "What's the most interesting part?"

Lead with the swing-geometry feature — newest, and the most range packed into
one piece of work.

"A hitter's stance barely changes from pitch to pitch, but where they actually
make contact does — and that gap is basically their timing on display. I built
a feature that shows this: a top-down map of a hitter's stance and swing
contact points, split out by pitch type. It pulls data from three different
Statcast/Savant sources that don't have existing wrappers, so I wrote my own
loaders for them, joined them together with SQL, and designed the chart from
scratch."

## Stories worth telling (the "why," not just the "what")

These are the answers that show judgment, not just tool names.

**Why SQL, and why DuckDB specifically.**
"I needed to combine data from separate sources — swing mechanics, stance
data, player heights — and rank hitters against the league. Rather than
standing up a full database server for a personal project, I used DuckDB,
which runs SQL directly against data already sitting in memory. It let me use
real SQL — joins, window functions, grouping — with zero infrastructure
overhead."

**Height-adjusted comparisons.**
"A raw number like '39-inch stance' doesn't mean anything on its own. I
divided stance width by the player's height before ranking it against the
league, so a 6'6\" and a 5'9\" hitter get compared fairly. That's the
difference between reporting a number and reporting an insight."

**Graceful degradation.**
"Not every player has complete tracking data — some just don't have enough
playing time to qualify for Savant's stance dataset. Instead of the report
silently breaking or showing nothing, I built it to explain *why* a section is
missing when it is. That was a deliberate choice, not an afterthought."

**Catching my own mistake.**
"I was about to publish a public write-up with a specific claim about a
player's contact points. Before posting, I went back and verified the numbers
against a second player — and the direction I'd described was actually
backwards from what the data showed. I only caught it because I double-checked
instead of trusting my first read. That's become a habit for anything that
goes public."

## "What did you learn building this?"

It's a genuine learning project — fine to say that directly:

"I used this project specifically to learn the engineering side, not just to
get something working. SQL was the big one — I hadn't used it seriously
before this, and now I can talk through window functions, joins, and CTEs, not
just recite syntax. I also learned the difference between 'a chart that's
technically correct' and 'a chart that actually reads clearly' — the
visualization went through a lot of real iteration before it looked right."

## If they probe on SQL specifically

Explain the concept in plain language — you don't need to write SQL live:

- **"What's a window function?"** → "A way to rank or compare rows against a
  group without collapsing them into a summary — every row keeps its identity,
  but gains a computed value, like a percentile."
- **"Difference between an inner and a left join?"** → "An inner join only
  keeps rows that match in both tables. A left join keeps everything from the
  first table, and leaves blanks where the second table has no match. I used a
  left join deliberately in one place, specifically so players missing from
  one dataset wouldn't disappear from the report entirely."
- **"Why use a CTE?"** → "It lets you build a query in named, readable steps
  instead of one nested block — and in my case it mattered for correctness
  too: I needed to rank one table *before* joining it to another, so the
  ranking wasn't skewed by players missing from the second table."

Full worked examples of all of this are in `docs/sql-notes.md` if you want to
refresh the details before an interview.

## Words you don't want to stumble on

- **DuckDB** — "duck-D-B," an in-process SQL database
- **Statcast** — MLB's pitch/swing tracking system
- **Baseball Savant** — the public site that exposes Statcast data
- **CUME_DIST** — don't say this out loud; describe it as "a percentile
  ranking function" instead
