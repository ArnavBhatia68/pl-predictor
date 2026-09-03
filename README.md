# PL Predictor

A leak-free machine-learning pipeline that predicts Premier League matches as
**home win / draw / away win probabilities** using detailed historical match
statistics from 2000/01 onward.

## What the production system does

1. Downloads each Premier League `E0.csv` season from Football-Data.co.uk.
   Finished seasons are validated at 380 matches; an open-data GitHub mirror is used if the
   primary archive serves an incomplete file.
2. Normalizes results, shots, shots on target, corners, fouls, and cards.
3. Replays matches chronologically and creates every row **before** updating team state.
4. Carries last-5, last-10, and venue form across adjacent seasons while smoothing early
   season-to-date form with a five-match prior from the preceding season.
5. Compares Logistic Regression, XGBoost, and Dixon-Coles Poisson models with time-based splits.
6. Selects the blend with the lowest validation log loss, evaluates it once on a held-out season,
   then refits the production components through the latest completed season.
7. Trains separate home/away count models for shots, shots on target, corners, fouls, and yellow
   cards, and saves their held-out MAE/RMSE reports.

No same-match statistics, future matches, team-name shortcuts, or bookmaker odds are used as
features.

## Project structure

```text
src/pl_predictor/
  data.py       download and normalize season CSVs
  elo.py        chronological Elo ratings
  features.py   leak-free rolling feature engine
  train.py      time-based model selection and evaluation
  fixtures.py   upcoming-fixture provider adapter
  tracking.py   SQLite prediction ledger and result grading
  api.py        FastAPI application
  cli.py        command-line entry point
tests/          leakage and Elo correctness tests
data/           generated raw and processed data
models/         trained model artifact
reports/        metrics and backtest predictions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,api]"
```

## Run the full V1 pipeline

The season argument is the year in which the season begins. For example, `2000`
means 2000/01 and `2026` means 2026/27.

```bash
pl-predictor download --start-season 2000 --end-season 2026
pl-predictor features
pl-predictor train \
  --validation-season 2023 \
  --test-seasons 2024 2025 \
  --exclude-seasons 2026
```

Run the stricter V2 tournament with:

```bash
pl-predictor v2 \
  --first-fold 2018 \
  --last-fold 2023 \
  --calibration-season 2024 \
  --test-season 2025
```

Run the goals-based V3 tournament with:

```bash
pl-predictor v3 \
  --first-fold 2018 \
  --last-fold 2023 \
  --calibration-season 2024 \
  --test-season 2025
```

Run the nested V2/V3 ensemble selection with:

```bash
pl-predictor v4 \
  --first-fold 2018 \
  --last-fold 2023 \
  --test-season 2025
```

Outputs:

```text
data/processed/matches.csv
data/processed/features.csv
models/best_model.joblib
reports/metrics.json
reports/test_predictions.csv
models/v2_model.joblib
reports/v2_metrics.json
reports/v2_walk_forward.csv
reports/v2_test_predictions.csv
reports/v2_feature_importance.csv
models/v3_poisson_model.joblib
reports/v3_metrics.json
reports/v3_walk_forward.csv
reports/v3_test_predictions.csv
reports/v3_feature_importance.csv
models/v4_ensemble_model.joblib
reports/v4_metrics.json
reports/v4_walk_forward.csv
reports/v4_test_predictions.csv
models/v11_stat_models.joblib
reports/v11_stat_metrics.json
reports/v11_stat_walk_forward.csv
reports/v11_stat_test_predictions.csv
```

## Verified V1 result (September 2, 2026)

The checked-in pipeline was run end-to-end against the complete data available at that time:

| Item | Result |
|---|---:|
| Completed historical matches, 2000/01–2025/26 | 9,880 |
| Current 2026/27 matches downloaded | 20 |
| Engineered pre-match features | 243 |
| Validation season | 2023/24 |
| Held-out test seasons | 2024/25 and 2025/26 |
| Selected V1 model | Logistic Regression |
| Test accuracy | 50.7% |
| Test log loss | 1.024 |
| Test macro F1 | 0.393 |

These are honest baseline numbers, not final project claims. The test set was not used to select
features or choose the model. Draw recall is currently weak, which gives V2 a concrete target:
walk-forward validation, tuned XGBoost, class-aware objectives, and probability calibration.

## Verified V2 experiment

V2 uses a stricter expanding-window evaluation:

```text
select model: 2018/19 -> 2023/24 walk-forward folds
calibrate:    2024/25
final test:   2025/26 (untouched until model selection is complete)
```

Eight configurations compared Logistic Regression and XGBoost, full versus compact features,
five-year time decay, and a modest draw weight. The winner was shallow XGBoost using 63 compact
features.

| Metric | V2 result |
|---|---:|
| Mean walk-forward accuracy | 55.0% |
| Mean walk-forward log loss | 0.967 |
| Final 2025/26 accuracy | 48.2% |
| Final 2025/26 log loss | 1.030 |
| Final 2025/26 macro F1 | 0.364 |

On the same 2025/26 test season, V1 recorded 49.2% accuracy and 1.047 log loss. V2 therefore
improved probability quality but not hard-class accuracy. The draw-weighted candidates improved
draw recall in walk-forward validation while worsening log loss, so they were not selected.

Global SHAP importance is written to `reports/v2_feature_importance.csv`; Elo difference was the
strongest V2 feature, followed by season shot dominance and the two teams' individual Elo ratings.

## Verified V3 Poisson experiment

V3 predicts two positive goal rates before each match:

```text
home expected goals = lambda_home
away expected goals = lambda_away
              -> Poisson scoreline matrix
              -> away / draw / home probabilities
```

It compares linear Poisson regression, histogram gradient boosting with Poisson loss, and XGBoost
Poisson regression, both with and without five-year time decay. It also compares five Dixon-Coles
low-score corrections. Selection uses the same six expanding walk-forward seasons as V2.

The winner was regularized linear Poisson regression with 155 goal-model features, no time decay,
and Dixon-Coles `rho=0.05`.

| Metric | V2 | V3 Poisson |
|---|---:|---:|
| Mean walk-forward log loss | 0.9671 | **0.9614** |
| Final 2025/26 accuracy | 48.2% | 48.2% |
| Final 2025/26 log loss | 1.02975 | **1.02956** |
| Final Brier score | 0.61934 | **0.61870** |
| Expected calibration error | 5.43% | **3.48%** |
| Mean goal MAE | — | 0.888 |

The pipeline performs a nested chronological calibration decision: train before 2023/24, fit the
calibrator on 2023/24, and check it on 2024/25. Calibration worsened 2024/25 log loss from 0.986 to
0.989, so the final production artifact correctly retains the raw Poisson probabilities.

Poisson provides useful draw probability without necessarily selecting draw as the largest class.
In the final season, the production model's average draw probability was 23.1% versus a 27.4%
actual draw rate. A separately calibrated variant moved the mean draw probability to 26.3%, but
worsened total log loss and had already failed the chronological calibration check.

V3 also outputs each fixture's expected goals and most likely scoreline. Global home-goal and
away-goal SHAP rankings are written to `reports/v3_feature_importance.csv`.

## Verified V4 ensemble experiment

V4 combines the probability distributions from the calibrated V2 classifier and raw V3 Poisson
goal model. Every walk-forward fold reconstructs production timing:

```text
V2 base classifier: train before the prior season
V2 calibrator:      fit on the prior season
V3 Poisson model:   train on every season before the prediction season
Ensemble:           predict the next season
```

Twenty-one blend weights from 0% to 100% classifier contribution were evaluated across the six
walk-forward seasons. The selected blend was **25% V2 classifier + 75% V3 Poisson**.

| Model | 2025/26 accuracy | 2025/26 log loss | Macro F1 |
|---|---:|---:|---:|
| V1 Logistic Regression | **49.2%** | 1.04662 | **0.379** |
| V2 calibrated XGBoost | 48.2% | 1.02975 | 0.364 |
| V3 linear Poisson | 48.2% | 1.02956 | 0.354 |
| V4 ensemble | 48.9% | **1.02766** | 0.366 |

V4 achieved 0.96049 mean walk-forward log loss, improving on V3's 0.9614 and V2's 0.9671.
The two components disagreed on their argmax outcome for 8.4% of final-season fixtures, providing
enough complementary information for blending to help. V4 still does not select draws as the
largest class, but assigns an average 23.6% draw probability against a 27.4% actual draw rate.

The original V1 retains the highest hard-class accuracy by 0.3 percentage points, while V4 has the
best probability quality. Because the application displays win/draw/loss probabilities, log loss
is the primary model-selection metric.

## V5 live prediction API

V5 adds a chronological live-state engine that replays every completed match and generates the
same feature schema for an unplayed fixture. A regression test proves that replaying to a known
historical fixture reproduces its original offline pre-match features exactly.

Before starting the backend, create or refresh the combined match dataset:

```bash
pl-predictor download \
  --start-season 2000 \
  --end-season 2026 \
  --refresh-current
```

`--refresh-current` redownloads only 2026/27 while reusing the cached historical seasons. Restart
the API after refreshing so its in-memory feature state includes the latest results.

Start FastAPI:

```bash
python -m uvicorn pl_predictor.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model, season, and data-freshness status |
| `GET /teams` | Current active-season clubs |
| `POST /predict` | V4 ensemble prediction for an upcoming fixture |
| `GET /model-performance` | Held-out model metrics and ensemble weights |
| `GET /historical-predictions` | Stored 2025/26 backtest predictions |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"home_team":"Chelsea","away_team":"Arsenal"}'
```

The response includes ensemble probabilities, component probabilities, expected goals, most likely
scoreline, current Elo, last-five form, confidence, and the latest included match date. Common team
aliases such as `Manchester City`, `Manchester United`, `Nottingham Forest`, and `Spurs` are
resolved to the historical dataset's canonical names.

## V6 fixture sync and prediction tracking

V6 connects the live model to football-data.org's Premier League fixture endpoint and stores an
immutable prediction ledger in `state/pl_predictor.sqlite3`. Register for an API token, keep it in
the server environment, and never expose it to the frontend:

```bash
export FOOTBALL_DATA_API_TOKEN="your-token"
```

Before syncing, refresh the detailed match dataset and restart the API:

```bash
pl-predictor download \
  --start-season 2000 \
  --end-season 2026 \
  --refresh-current

pl-predictor sync-fixtures --days-back 3 --days-ahead 14
```

Each sync performs four operations:

1. Upserts scheduled and completed fixtures using the provider's stable fixture ID.
2. Publishes predictions only for the next matchweek when its first kickoff is within three days.
3. Never overwrites that pre-match prediction on later syncs.
4. Grades completed fixtures for correctness, multiclass log loss, and Brier score.

If the provider reports a completed match newer than `data/processed/matches.csv`, V6 still stores
and grades the result but blocks new predictions. Refreshing the detailed dataset and restarting
the process clears that safety lock. This prevents an upcoming prediction from quietly ignoring a
team's latest shots, goals, cards, and Elo update.

The additional API endpoints are:

| Endpoint | Purpose |
|---|---|
| `POST /fixtures/sync` | Fetch fixtures, save new predictions, and grade results |
| `GET /fixtures/upcoming` | Upcoming fixtures with stored model probabilities |
| `GET /live-predictions` | Auditable pre-match prediction history |
| `GET /season-record` | Live accuracy, log loss, and Brier score |

`POST /fixtures/sync` is convenient for local development. Before a public deployment, invoke the
same CLI command from a private scheduled job or protect the endpoint with server-side
authentication so visitors cannot consume the fixture provider's rate limit.

## V7 gameweek dashboard

V7 adds the production frontend in `frontend/`. It is a responsive Next.js/Vinext dashboard with
three working surfaces:

- **Gameweek** selects an upcoming fixture and displays its locked outcome probabilities,
  expected goals, likely scoreline, and model pick.
- **History** shows the immutable pre-match ledger, final scores, correctness, and log loss.
- **Model** shows the live season record and the 75% Poisson / 25% calibrated XGBoost ensemble.

Connect it to FastAPI with:

```bash
cd frontend
cp .env.example .env.local
npm run install:ci
npm run dev
```

`NEXT_PUBLIC_API_BASE_URL` should contain the FastAPI origin, such as `http://localhost:8000`.
Without that variable, the dashboard shows a data-feed error and never substitutes invented
numbers. When the frontend and backend use different origins, add the frontend origin to
`CORS_ORIGINS` before starting FastAPI.

The current 2026/27 season is downloaded and used to update current team state later, but it is
excluded from V1 model evaluation because it is incomplete.

## V8 production deployment

V8 packages FastAPI and the trained ensemble as a single Docker service. It also closes the two
production gaps from V7:

- `POST /fixtures/sync` and `POST /admin/refresh` require the `X-Sync-Key` header.
- The deployed process can refresh the current Football-Data.co.uk season, rebuild live team
  state, sync upcoming football-data.org fixtures, and grade completed predictions on a schedule.

Required server variables:

| Variable | Purpose |
|---|---|
| `FOOTBALL_DATA_API_TOKEN` | Server-only football-data.org token |
| `SYNC_API_KEY` | Secret for protected refresh endpoints |
| `AUTO_REFRESH_INTERVAL_HOURS` | Refresh cadence; `0` disables the background loop |
| `PREDICTION_PUBLISH_LEAD_DAYS` | Days before the next matchweek when official picks are locked |
| `PL_PREDICTOR_DB_PATH` | SQLite ledger path |

Run the production image locally with:

```bash
docker build -t pl-predictor-api .
docker run --rm -p 8000:8000 \
  -e FOOTBALL_DATA_API_TOKEN="your-token" \
  -e SYNC_API_KEY="your-random-secret" \
  -e AUTO_REFRESH_INTERVAL_HOURS=6 \
  pl-predictor-api
```

Trigger a protected immediate refresh with:

```bash
curl -X POST http://localhost:8000/admin/refresh \
  -H 'Content-Type: application/json' \
  -H 'X-Sync-Key: your-random-secret' \
  -d '{"days_back":3,"days_ahead":14}'
```

`render.yaml` describes the Render web service. The local SQLite ledger is still ephemeral on the
free instance; V9 adds an automated public-data snapshot and restore layer around it.

## V9 matchweek intelligence and reliability

V9 turns the dashboard into a full matchweek intelligence surface:

- current Premier League table, points per game, goal difference, and recent form;
- all fixtures in the next matchweek and the previous matchweek's prediction review;
- full match centres with last-10 form, historical head-to-head, outcome probabilities,
  expected goals, and forecasts for shots, shots on target, corners, fouls, cards, and red-card
  probability;
- one profile for every active team with its last 10 performances and stored model calls;
- an immutable prediction history mirrored into the dashboard's managed D1 database.

The backend refreshes the detailed current-season match file before it creates new predictions.
The scheduled GitHub workflow wakes the sleeping free Render instance every six hours, waits for
the refresh to finish, then writes the non-sensitive prediction ledger to the `live-data` branch.
On a fresh container, the API restores that ledger before syncing fixtures, so a restart cannot
silently replace or erase earlier pre-match probabilities.

Possession is deliberately not forecast because the historical training source does not contain
possession labels. Player watch lists use football-data.org's scorer endpoint when the connected
plan exposes it; otherwise the API returns an explicit availability reason instead of fabricating
players or statistics.

## V11 standardized forecasting upgrade

V11 fixes early-season fragility and makes every displayed forecast auditable:

- last-5, last-10, and venue form cross an adjacent season boundary, so Gameweek 3 can use the
  two current matches plus the most recent matches from the preceding campaign;
- season-to-date features reset normally but are blended with a five-match prior. Returning clubs
  use their preceding PL season and promoted clubs use the preceding league average;
- the equal-weight historical training candidate beat the five-year time-decay candidate, so old
  training rows remain available without an arbitrary recency multiplier;
- the production outcome artifact is refit through 2025/26 after the untouched test report is
  generated;
- the displayed expected-goal rates, modal scoreline, and top-five scorelines all come from the
  same Dixon-Coles score matrix;
- separate Poisson count models forecast shots, shots on target, corners, fouls, and yellow cards.
  Serving enforces `shots_on_target <= shots` and returns 80% count intervals;
- official predictions remain immutable. When a model changes during an open matchweek, its new
  forecasts are stored as shadow predictions and never counted in the official record.

The corrected six-season walk-forward run selected **15% XGBoost + 85% Dixon-Coles Poisson** with
0.9594 mean validation log loss. On the untouched 2025/26 season it recorded 48.2% outcome
accuracy and 1.0246 log loss. Detailed-stat held-out MAE was 3.36 shots, 1.77 shots on target,
2.13 corners, 2.77 fouls, and 1.01 yellow cards per team-match.

Additional V9 endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /dashboard` | Matchweek board, previous results, table, record, and model status |
| `GET /fixtures/{fixture_key}` | Match centre, H2H, team form, and stat forecasts |
| `GET /teams/{team}` | Team profile, last 10 performances, and prediction history |
| `GET /refresh-status` | Scheduled refresh readiness and last result |

## V10 official publication and season-long evaluation

V10 prevents distant fixtures from being locked with stale form. The schedule remains visible,
but an official prediction is created only for the immediately upcoming matchweek once its first
kickoff is within the configured three-day window. Any old, premature future predictions are
retired and regenerated when their matchweek becomes official. Once published in the valid
window, the probability remains immutable.

The prediction ledger now also stores the pre-match shots, shots-on-target, corners, fouls, and
yellow-card forecasts. After the detailed result arrives, it records the observed values and
generates a deterministic statistical review explaining the largest misses without an AI service.
The dashboard exposes cumulative season history, per-team accuracy, log loss, Brier score, and
goal error. Its managed database keeps every graded prediction rather than limiting history to a
fixed number of matchweeks.

The current season start is determined automatically from the calendar. At a season boundary,
the live state regresses Elo ratings, resets season form, and can register the new fixture list
before the first completed result exists, so no annual environment-variable edit is required.

## Why the chronology matters

For each match, the feature engine does this:

```text
read fixture
  -> calculate features from prior matches only
  -> store features and actual result
  -> update both teams with the completed match
```

If Chelsea take 18 shots against Arsenal, those 18 shots cannot help predict that same match.
They only affect Chelsea's and Arsenal's next fixtures. The tests enforce this rule.

## Evaluation

V1 reports:

- accuracy
- macro F1
- multiclass log loss
- multiclass Brier score
- confusion matrix
- per-class precision, recall, and F1

Log loss is the model-selection metric because the product displays probabilities, not only the
most likely class.

Run the dependency-free core correctness tests with:

```bash
python -m unittest discover -s tests -v
```

## Next milestones

1. Add rest days, promoted-team initialization, managerial changes, and squad-availability data.
2. Compare against bookmaker implied probabilities as a benchmark, never as training inputs.
3. Add rolling-origin probability calibration across multiple seasons.
4. Add a richer provider for historical possession, lineups, injuries, and player-level form.
5. Compare the descriptive stat forecasts with independently trained count regressors.

Data source: [Football-Data.co.uk](https://www.football-data.co.uk/data.php). Credit the source
when publishing analysis or derived datasets.
