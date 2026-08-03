# SQL Notes — a personal reference

My working notes on the SQL used in this project's swing-geometry feature
(`src/swing_geometry.py`). Every example is real code from the three queries we
wrote, so this doubles as a walkthrough of that module.

We run SQL with **DuckDB**, an in-process engine that queries pandas DataFrames
directly — no database server. In code it looks like:

```python
con = duckdb.connect()
con.register("swings", swings_df)     # expose a pandas frame as a SQL table
result = con.execute(SQL, [param]).df()   # run SQL, get a pandas frame back
```

The three queries this references:
- **Query A** — per-pitch breakdown (`_BY_PITCH_SQL`)
- **Query B** — league percentiles (`_LEAGUE_PERCENTILE_SQL`)
- **Query C** — height-adjusted percentiles (`_STANCE_BIOMECH_SQL`)

---

## Quick reference

| Keyword / function | One-line meaning |
|---|---|
| `SELECT` | Which columns to return |
| `FROM` | Which table to read |
| `WHERE` | Keep rows matching a condition (before grouping) |
| `GROUP BY` | Collapse rows into one summary row per group |
| `HAVING` | Filter groups (after grouping) — like WHERE but for aggregates |
| `ORDER BY … DESC` | Sort output rows (DESC = largest first) |
| `COUNT(*)` / `AVG(x)` / `ROUND(x,1)` | Count rows / average a column / round |
| `FUNC() OVER (ORDER BY x)` | **Window function** — compute across rows but keep every row |
| `CUME_DIST()` | Fraction of rows at or below this one = a percentile (0–1) |
| `JOIN … ON` | Combine two tables on a matching key |
| `INNER JOIN` | Keep only rows that match in **both** tables |
| `LEFT JOIN` | Keep **all** left-table rows; blanks where the right has no match |
| `WITH name AS (…)` | A named temporary result (CTE) to build the query in steps |
| `IS NOT NULL` | Keep rows where a value actually exists |
| `?` | A placeholder value filled in safely at run time |

---

## 1. The skeleton: SELECT / FROM / WHERE

```sql
SELECT pitch_type, bat_speed
FROM swings
WHERE attack_angle IS NOT NULL
```
"From `swings`, keep rows that have an attack angle, return pitch type + bat speed."

- `FROM` = the source table.
- `WHERE` = row filter, applied **before** any grouping.
- `SELECT` = the columns you want.

`SELECT *` = all columns. `sw.*` = all columns from the table aliased `sw`.

---

## 2. Aggregate functions + GROUP BY (Query A)

Aggregates squash many rows into one number: `COUNT(*)`, `AVG(x)`, `MIN`, `MAX`,
`SUM`. `GROUP BY` makes them run **per group** instead of over the whole table.

```sql
SELECT
    pitch_type,
    COUNT(*)                      AS swings,
    ROUND(AVG(attack_angle), 1)   AS attack_angle
FROM swings
WHERE attack_angle IS NOT NULL    -- drop non-swings (no tracking)
GROUP BY pitch_type               -- one row per pitch type
HAVING COUNT(*) >= 15             -- drop pitch types barely seen
ORDER BY swings DESC              -- most-seen first
```

- **`GROUP BY pitch_type`** → one summary row per distinct pitch type; `COUNT`/`AVG`
  are now computed within each type.
- **`HAVING`** filters *groups* (runs after grouping). `WHERE` filters *rows*
  (runs before). Rule: condition on a raw column → `WHERE`; condition on an
  aggregate like `COUNT(*)` → `HAVING`.
- **`ORDER BY … DESC`** sorts the output.

---

## 3. Window functions (Query B)

A window function computes across a set of rows **but keeps every row** (unlike
GROUP BY, which collapses them). Syntax: `FUNCTION() OVER (ORDER BY column)`.
The `OVER (…)` is what makes it a window function.

```sql
SELECT name,
       CUME_DIST() OVER (ORDER BY avg_bat_speed) AS bat_speed_pct
FROM league_swing_path
```

- **`CUME_DIST()`** = the fraction of rows at or below this one → a percentile.
  0.96 means "faster than 96% of the league."
- Every hitter stays a row; each just gains a ranking column.

In Query B we stack five of these in one pass — one per metric — so a single
query ranks the whole league on everything at once.

**Relatives (same `OVER` syntax):** `RANK()` (1,2,3 with ties), `ROW_NUMBER()`
(unique 1,2,3), `PERCENT_RANK()` (0-based percentile), `AVG(x) OVER (…)`
(running/group average that keeps rows).

---

## 4. Joins: INNER vs LEFT (the one I keep mixing up)

A join combines two tables by matching a key. Table **aliases** (`s`, `st`, `h`)
are short nicknames; **`ON`** is the matching rule.

### Concrete demo

`swings` has 3 hitters; `stance` is missing Judge (as in real Savant data):

```
swings                    stance (no Judge)
 id  name   bat_speed      id  foot_sep
101  Judge   77.0          102   8.0
102  Kwan    68.0          103  43.0
103  Soto    74.0
```

```sql
-- INNER JOIN: only ids in BOTH tables
SELECT s.id, s.name, st.foot_sep
FROM swings s JOIN stance st ON s.id = st.id;
```
```
 id name  foot_sep
102 Kwan       8.0
103 Soto      43.0        <- Judge DROPPED (no stance row)
```

```sql
-- LEFT JOIN: ALL swings rows, stance where it exists
SELECT s.id, s.name, st.foot_sep
FROM swings s LEFT JOIN stance st ON s.id = st.id;
```
```
 id  name  foot_sep
102  Kwan       8.0
103  Soto      43.0
101 Judge       NaN        <- Judge KEPT, stance blank (NULL)
```

### The mental model
Ask: **"which table's rows are guaranteed to survive?"**
- **INNER** = the *intersection*. A row survives only if the key exists in
  **both** tables. Non-matches disappear.
- **LEFT** = *all of the left table*, plus matches from the right. Left rows
  always survive; unmatched right columns come back **NULL/NaN**.

"Left" / "right" = the tables' positions in the query:
`FROM swings LEFT JOIN stance` → `swings` is left (protected), `stance` is right
(optional).

### Rule of thumb
Do I want unmatched rows to **disappear** or **stay with blanks**?
- disappear → **INNER**
- stay → **LEFT**

### In our code
- **Query B:** `swing_ranked sw LEFT JOIN stance_ranked st ON sw.id = st.id AND
  sw.side = st.side` — **LEFT** on purpose, so a hitter with no stance row still
  appears (feet blank, disclaimer shown) instead of vanishing from the report.
  The `ON … AND …` matches on **two** keys (id + batting side) for switch hitters.
- **Query C:** `league_stance s JOIN league_heights h ON s.id = h.id` — plain
  **INNER**, because a hitter with no height can't be ranked by height anyway, so
  dropping him is correct.

> Memory hook: **INNER = only the overlap. LEFT = all of the left, plus overlap;
> non-matches come back blank.**

---

## 5. CTEs — WITH name AS (…) (Query C)

A CTE is a **named temporary result** defined up top and used below — like a
variable for a sub-query. Keeps complex queries readable and lets you build in
steps; you can chain several.

```sql
WITH joined AS (                       -- step 1: stance + heights, make the ratio
    SELECT s.id, s.avg_foot_sep, h.height_in,
           s.avg_foot_sep / h.height_in AS width_ratio
    FROM league_stance s
    JOIN league_heights h ON s.id = h.id
    WHERE h.height_in IS NOT NULL AND s.avg_foot_sep IS NOT NULL
),
ranked AS (                            -- step 2: percentile over step 1
    SELECT *, CUME_DIST() OVER (ORDER BY width_ratio) AS width_pct
    FROM joined
)
SELECT * FROM ranked WHERE id = ?;     -- step 3: pull one player out
```

**Why CTEs mattered in Query B:** we rank each table *inside its own CTE*
(`swing_ranked`, `stance_ranked`) **before** joining — so a hitter's stance
percentile is measured only against hitters who *have* stance data, not diluted
by the ones who don't.

---

## 6. Computed columns, NULLs, parameters

**Arithmetic in SELECT** — make a new column on the fly; `AS` names it:
```sql
s.avg_foot_sep / h.height_in AS width_ratio
```
This is the trick behind "wide *for his size*": build the ratio, then rank the
ratio — so tall and short hitters compare fairly.

**NULL handling** — `NULL` = "no value"; test it with `IS NULL` / `IS NOT NULL`
(never `= NULL`). `AND` chains conditions:
```sql
WHERE h.height_in IS NOT NULL AND s.avg_foot_sep IS NOT NULL
```

**Parameters (`?`)** — a placeholder filled in at run time, kept separate from the
query text so a value can't be misread as SQL (prevents "SQL injection"):
```python
con.execute(_LEAGUE_PERCENTILE_SQL, [player_id])   # ? becomes player_id
```

---

## 7. How to read any query fast — execution order

SQL is *written* `SELECT … FROM … WHERE …`, but it *runs* in this order. Reading
it this way makes any query click:

1. `FROM` / `JOIN` — assemble source rows
2. `WHERE` — filter rows
3. `GROUP BY` — bucket them
4. `HAVING` — filter buckets
5. window functions (`OVER`) — add ranking columns
6. `SELECT` — pick / compute final columns
7. `ORDER BY` — sort output

---

## 8. The three real queries (for reference)

- **Query A — `_BY_PITCH_SQL`**: `GROUP BY` + `HAVING` — collapse ~1,500 swings
  into one row per pitch type with averages.
- **Query B — `_LEAGUE_PERCENTILE_SQL`**: two CTEs each using `CUME_DIST`
  window functions, joined with a `LEFT JOIN` on (id, side).
- **Query C — `_STANCE_BIOMECH_SQL`**: an `INNER JOIN` (stance + heights) inside
  a CTE, a computed ratio column, then `CUME_DIST` percentiles on the ratio.

All three live in `src/swing_geometry.py`.
