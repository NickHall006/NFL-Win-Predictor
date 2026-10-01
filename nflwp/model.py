"""Models, honest evaluation, and per-prediction explanations."""
import itertools
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_DIR, TRAIN_START
from .features import FEATURES, GROUPS

MODEL_NAMES = ["logistic", "forest", "boosting", "ensemble"]


# --- models --------------------------------------------------------------------
class AverageEnsemble:
    """Averages predicted probabilities of several models."""
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))

def _forest():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=30, max_features=0.5,
                                  n_jobs=-1, random_state=0)

def _boosting():
    return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=120, max_depth=3,
                                          min_samples_leaf=40, l2_regularization=5.0,
                                          random_state=0)

def make_model(name: str):
    if name == "logistic": return _logistic()
    if name == "forest":   return _forest()
    if name == "boosting": return _boosting()
    if name == "ensemble": return AverageEnsemble([_logistic(), _forest(), _boosting()])
    raise ValueError(f"unknown model '{name}' (choose from {MODEL_NAMES})")


def training_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Completed true-home games (with play-by-play) from TRAIN_START on; ties and
    neutral-site games dropped."""
    played = games["h_played"].fillna(False).astype(bool) & games["a_played"].fillna(False).astype(bool)
    m = (games["season"] >= TRAIN_START) & games["home_win"].notna() & played & (games["location"] == "Home")
    return games[m]


def _xy(df):
    return df[FEATURES].fillna(0.0), df["home_win"].astype(int)


# --- backtest ------------------------------------------------------------------
def _vegas_prob(spread, sd=13.5):
    """Convert a point spread (home favored = positive) to a home win probability."""
    return 0.5 * (1 + np.vectorize(math.erf)(spread / sd / math.sqrt(2)))


def walk_forward(games: pd.DataFrame, first_test: int, last_test: int) -> pd.DataFrame:
    """For each season S: train on seasons < S, predict season S. No peeking."""
    data = training_frame(games)
    out = []
    for season in range(first_test, last_test + 1):
        train, test = data[data.season < season], data[data.season == season]
        if test.empty:
            continue
        Xtr, ytr = _xy(train)
        Xte, yte = _xy(test)
        res = test[["game_id", "season", "week", "home_team", "away_team", "spread_line"]].copy()
        res["y"] = yte.to_numpy()
        res["always_home"] = ytr.mean()
        elo_only = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr[["d_elo"]], ytr)
        res["elo_only"] = elo_only.predict_proba(Xte[["d_elo"]])[:, 1]
        for name in MODEL_NAMES:
            res[name] = make_model(name).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        res["vegas"] = _vegas_prob(res["spread_line"]) if res["spread_line"].notna().all() else np.nan
        out.append(res)
        print(f"  backtested {season}")
    return pd.concat(out, ignore_index=True)


def score_table(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["always_home", "elo_only", "vegas"] + MODEL_NAMES:
        d = bt.dropna(subset=[name])
        p = d[name].clip(0.01, 0.99)
        rows.append({"model": name, "n": len(d),
                     "accuracy": accuracy_score(d["y"], p > 0.5),
                     "log_loss": log_loss(d["y"], p),
                     "brier": brier_score_loss(d["y"], p)})
    return pd.DataFrame(rows).set_index("model")


def calibration_table(y, p, bins=(0, .3, .4, .5, .6, .7, .8, 1.0)) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["bucket"] = pd.cut(d["p"], bins=list(bins), include_lowest=True)
    t = d.groupby("bucket", observed=True).agg(games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return t


def global_importance(games: pd.DataFrame, holdout_season: int) -> pd.DataFrame:
    """Permutation importance (drop in log-loss when a feature is shuffled) for the
    boosted model trained on seasons < holdout_season, measured on holdout_season."""
    data = training_frame(games)
    Xtr, ytr = _xy(data[data.season < holdout_season])
    Xte, yte = _xy(data[data.season == holdout_season])
    m = _boosting().fit(Xtr, ytr)
    r = permutation_importance(m, Xte, yte, scoring="neg_log_loss", n_repeats=20, random_state=0)
    lr = _logistic().fit(Xtr, ytr)
    coefs = pd.Series(lr[-1].coef_[0], index=FEATURES)
    return pd.DataFrame({"perm_importance_logloss": r.importances_mean,
                         "logistic_std_coef": coefs}, index=FEATURES).sort_values(
        "perm_importance_logloss", ascending=False)


# --- train / save / load -------------------------------------------------------
def train_and_save(games: pd.DataFrame) -> None:
    X, y = _xy(training_frame(games))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        joblib.dump(make_model(name).fit(X, y), MODEL_DIR / f"{name}.joblib")
    print(f"  trained {len(MODEL_NAMES)} models on {len(X)} games")


def load_model(name: str):
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"No trained model '{name}'. Run:  python run.py train")
    return joblib.load(path)


# --- explanations --------------------------------------------------------------
class Symmetric:
    """Neutral-site wrapper: average 'A hosts B' with the mirror image 'B hosts A',
    which cancels the home edge. Works because every feature is a home-minus-away
    difference, so swapping teams just negates the features."""
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        p = 0.5 * (self.model.predict_proba(X)[:, 1] + 1 - self.model.predict_proba(-X)[:, 1])
        return np.column_stack([1 - p, p])


def predict_home_prob(model, X: pd.DataFrame, neutral: bool = False) -> np.ndarray:
    m = Symmetric(model) if neutral else model
    return m.predict_proba(X[FEATURES].fillna(0.0))[:, 1]


def explain(model, x: pd.Series, neutral: bool = False, groups=GROUPS):
    """Break a prediction into contributions from each feature group.

    Start: a coin flip (50%). Step 1 - HOME FIELD: how far above 50% the model puts two
    identical teams playing at home (0 at a neutral site). Step 2 - each remaining group's
    exact Shapley value: its average effect on the home win probability over every order
    in which the groups could be 'switched on' from identical teams. Model-agnostic, and
    contributions sum exactly to (final prob - 0.5).

    Returns (final_prob, {group: contribution in probability points}).
    """
    m = Symmetric(model) if neutral else model
    x = x[FEATURES].astype(float).fillna(0.0)
    names = list(groups)
    n = len(names)

    masks = list(itertools.product([0, 1], repeat=n))
    rows = []
    for mask in masks:
        r = pd.Series(0.0, index=FEATURES)
        for on, g in zip(mask, names):
            if on:
                r[groups[g]] = x[groups[g]]
        rows.append(r)
    probs = m.predict_proba(pd.DataFrame(rows)[FEATURES])[:, 1]
    value = dict(zip(masks, probs))

    fact = math.factorial
    phi = {"Home field": value[(0,) * n] - 0.5}
    for i, g in enumerate(names):
        total = 0.0
        for mask in masks:
            if mask[i]:
                continue
            s = sum(mask)
            w = fact(s) * fact(n - s - 1) / fact(n)
            total += w * (value[mask[:i] + (1,) + mask[i + 1:]] - value[mask])
        phi[g] = total
    return value[(1,) * n], phi
