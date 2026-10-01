# NFL Win Probability Model

Predicts the **probability** each team wins an NFL game, and **explains which factors drove it**.
Built with Python, pandas, and scikit-learn on [nflverse](https://github.com/nflverse) data.

```
BUF vs KC  ->  BUF 71.0% | KC 29.0%

  Team strength   +8.4 pp  favors BUF
  Quarterback     +7.1 pp  favors BUF
  Home field      +5.3 pp  favors BUF
  Defense         -0.5 pp  favors KC
  ...
```

## Quickstart

```bash
pip install -r requirements.txt

python run.py update      # download nflverse data (~1 min first time) + build features
python run.py backtest    # honest walk-forward evaluation
python run.py train       # fit models on all completed games
python run.py week        # win probabilities + drivers for the upcoming week
python run.py predict --home BUF --away KC
python run.py predict --home BUF --away KC --home-qb Darnold  # what-if: swap the QB
python run.py predict --home BUF --away KC --neutral          # neutral-site game
```

Re-run `update` then `train` each week to pick up new results.

## How it works

**Data** (`nflwp/data.py`) - `games.csv` (schedule, scores, rest days; includes future games) and
per-season play-by-play, trimmed to the columns we need and cached locally.

**Features** (`nflwp/features.py`) - all are *home minus away*, oriented so higher = better:

| Group | Features |
|---|---|
| Offense | EPA/play passing, EPA/play rushing (garbage time removed) |
| Defense | EPA/play allowed vs pass, vs rush |
| Quarterback | Starter's EPA/dropback, decayed over time and shrunk toward average so a 1-game wonder isn't trusted |
| Turnovers | Takeaways minus giveaways per game |
| Recent form | Net EPA/play over the last 3 games |
| Team strength | Elo rating, recent-weighted point differential |
| Rest | Home rest days minus away rest days |
| Home field | Modeled as the baseline (see below) |

**No leakage.** Rolling stats are computed in order and shifted one game, so a game only ever sees
information from *before* kickoff. Evaluation is walk-forward: to score season S, train only on seasons < S.

**Models** (`nflwp/model.py`) - logistic regression, random forest, gradient boosting, and an
average-of-all ensemble (default).

**Explanations** - each prediction is split into **exact Shapley values over feature groups**. Start at a 50% coin flip,
add the home-field edge, then measure how much each group (Offense, QB, ...) moves the probability, averaged
over every order the groups could be switched on. Contributions sum exactly to the final probability, and it
works for any model, including the tree ensembles.

**Home field** is deliberately not a feature. Models train on true home games only, so the home edge is the
model's baseline (~+5 pp for identical teams). Neutral-site games (London, Super Bowl) are predicted by averaging
"A hosts B" with the mirror image, which cancels the home edge.

## Results (walk-forward backtest, 2018-2025, 2,173 home games)

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Always pick home team | 54.4% | 0.6905 | 0.2486 |
| Elo only | 63.5% | 0.6369 | 0.2230 |
| Logistic regression | 65.0% | 0.6296 | 0.2199 |
| Random forest | 63.8% | 0.6318 | 0.2210 |
| Gradient boosting | 64.5% | 0.6301 | 0.2201 |
| **Ensemble** | 64.4% | **0.6286** | **0.2196** |
| Vegas closing line | 66.3% | 0.6097 | 0.2110 |

Read this honestly: NFL games are noisy, so ~65% accuracy is what good public models reach. The betting line
is the benchmark to chase, and it's still ahead because it knows about injuries, weather, and sharp money.
Accuracy differences between our models (~1 pt) are within noise for this sample; log loss and Brier
(which score the probabilities, not just the pick) are the better yardsticks. The model is well calibrated:
when it says 70-80%, the favorite wins about 71%.

What the model learned (`backtest` prints this): QB quality and overall team strength dominate; **turnover margin
carries almost no signal**, consistent with turnovers being largely luck.

## Ideas to extend it

- **Injuries / starting QB**: nflverse has injury and depth-chart data. Today the model assumes each team's most recent starter (use `--home-qb` to override).
- **Next Gen Stats**: pass-rush pressure, time to throw, and CPOE from [ngs-data](https://github.com/nflverse/ngs-data) as extra features.
- **Weather / roof / travel**: `games.csv` already has temp, wind, roof and surface.
- **Tune it**: half-lives (`features.py`), model hyperparameters (`model.py`). Do this with a validation split, not the backtest seasons.
- **Add the betting line as a feature**: the model gets much better, but at that point it's re-deriving the market rather than an independent estimate.
- **Simulate the season** by sampling each remaining game from its probability.

## Layout

```
run.py              CLI
nflwp/data.py       download + cache
nflwp/features.py   pandas feature engineering (rolling form, QB ratings, Elo)
nflwp/model.py      models, walk-forward backtest, Shapley explanations
```
