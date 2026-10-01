"""Turn raw play-by-play into pre-game features.

The golden rule: every feature for a game uses ONLY information available
before kickoff. We compute per-team "post-game" rolling states, then shift
them by one game so each game sees the state going *into* it.

Conventions
-----------
* Every stat is oriented so that HIGHER = BETTER (defense EPA is negated).
* Matchup features are HOME minus AWAY (`d_*`), so positive favors the home team.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CACHE

# --- feature definitions ---------------------------------------------------
STATE_COLS = [
    "off_epa_pass", "off_epa_rush",      # offensive EPA/play, by play type
    "def_epa_pass", "def_epa_rush",      # defensive EPA/play (negated allowed EPA)
    "turnover_margin",                   # takeaways - giveaways per game
    "qb_epa",                            # shrunk QB EPA/dropback of the starter
    "form_net_epa",                      # last-3-games net EPA/play (recent form)
    "elo", "pt_diff",                    # overall team strength
]
# Home-field advantage is deliberately NOT a feature: models are trained on true home
# games only, so home advantage is the model's baseline (see model.explain), and
# neutral-site games are handled by symmetrizing the prediction.
FEATURES = [f"d_{c}" for c in STATE_COLS] + ["rest_diff"]

# Used for the "what drove this prediction" breakdown
GROUPS = {
    "Team strength": ["d_elo", "d_pt_diff"],
    "Offense": ["d_off_epa_pass", "d_off_epa_rush"],
    "Defense": ["d_def_epa_pass", "d_def_epa_rush"],
    "Quarterback": ["d_qb_epa"],
    "Turnovers": ["d_turnover_margin"],
    "Recent form": ["d_form_net_epa"],
    "Rest": ["rest_diff"],
}

EWMA_COLS = ["off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush",
             "turnover_margin", "pt_diff"]
EWMA_HALFLIFE = 8        # games; ~ last 16 games carry most of the weight
FORM_WINDOW = 3          # games for "recent form"
QB_PRIOR_EPA = 0.0       # what an unknown QB is assumed to be worth
QB_PRIOR_DROPBACKS = 150 # shrinkage strength (~4 starts of evidence)
QB_HALFLIFE_GAMES = 20


@dataclass
class FeatureSet:
    games: pd.DataFrame    # one row per game with pre-game features + label
    latest: pd.DataFrame   # current state per team (for ad-hoc matchups)
    qbs: pd.DataFrame      # current rating per quarterback


# --- step 1: play-by-play -> per team-game stats ----------------------------
def _offense_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & (pbp["two_point_attempt"] != 1)].copy()
    not_garbage = p["wp"].between(0.05, 0.95)   # drop blowout garbage time from EPA
    p["epa_all"] = p["epa"].where(not_garbage)
    p["epa_pass"] = p["epa"].where(not_garbage & (p["pass"] == 1))
    p["epa_rush"] = p["epa"].where(not_garbage & (p["rush"] == 1))
    p["giveaway"] = ((p["interception"] == 1) |
                     ((p["fumble_lost"] == 1) & (p["fumbled_1_team"] == p["posteam"]))).astype(int)
    out = p.groupby(["game_id", "posteam"]).agg(
        off_epa=("epa_all", "mean"),
        off_epa_pass=("epa_pass", "mean"),
        off_epa_rush=("epa_rush", "mean"),
        giveaways=("giveaway", "sum"),
    ).reset_index()
    return out.rename(columns={"posteam": "team"})


def _qb_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["epa"].notna() &
            pbp["passer_player_id"].notna() & (pbp["two_point_attempt"] != 1)]
    out = d.groupby(["game_id", "posteam", "passer_player_id"]).agg(
        qb_name=("passer_player_name", "first"),
        db=("epa", "size"),
        epa_sum=("epa", "sum"),
    ).reset_index()
    return out.rename(columns={"posteam": "team", "passer_player_id": "qb_id"})


def _team_game_table(games: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (home & away view), with own + opponent stats."""
    def side(home: bool) -> pd.DataFrame:
        a, b = ("home", "away") if home else ("away", "home")
        return pd.DataFrame({
            "game_id": games["game_id"], "season": games["season"], "gameday": games["gameday"],
            "team": games[f"{a}_team"], "opp": games[f"{b}_team"],
            "pts_for": games[f"{a}_score"], "pts_against": games[f"{b}_score"],
        })

    base = pd.concat([side(True), side(False)], ignore_index=True)
    off = _offense_stats(pbp)
    opp = off.rename(columns={"team": "opp", "off_epa": "opp_off_epa",
                              "off_epa_pass": "opp_off_epa_pass",
                              "off_epa_rush": "opp_off_epa_rush",
                              "giveaways": "opp_giveaways"})
    tg = base.merge(off, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opp"], how="left")
    tg["def_epa_pass"] = -tg["opp_off_epa_pass"]
    tg["def_epa_rush"] = -tg["opp_off_epa_rush"]
    tg["net_epa"] = tg["off_epa"] - tg["opp_off_epa"]
    tg["turnover_margin"] = tg["opp_giveaways"] - tg["giveaways"]
    tg["pt_diff"] = tg["pts_for"] - tg["pts_against"]
    tg["played"] = tg["off_epa"].notna() & tg["opp_off_epa"].notna() & tg["pts_for"].notna()
    return tg.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# --- step 2: rolling team form (EWMA + last-3), shifted to be pre-game ------
def _rolling_states(tg: pd.DataFrame):
    played = tg[tg["played"]]
    grp = played.groupby("team")
    post = grp[EWMA_COLS].transform(lambda s: s.ewm(halflife=EWMA_HALFLIFE).mean())
    post["form_net_epa"] = grp["net_epa"].transform(lambda s: s.rolling(FORM_WINDOW, min_periods=1).mean())
    cols = EWMA_COLS + ["form_net_epa"]
    post["team"] = played["team"]

    pre = pd.DataFrame(index=tg.index, columns=cols, dtype=float)
    pre.loc[played.index, cols] = post.groupby("team")[cols].shift(1)       # state going INTO the game
    latest = post.groupby("team")[cols].last()                              # state after the latest game
    future = tg[~tg["played"]]
    pre.loc[future.index, cols] = latest.reindex(future["team"]).to_numpy()  # upcoming: use latest state
    return pre, latest


# --- step 3: quarterback ratings ---------------------------------------------
def _qb_ratings(qb: pd.DataFrame) -> pd.DataFrame:
    """Decayed, shrunk EPA/dropback per QB, computed chronologically.
    `qb_pre` is what we knew before that game; `qb_post` is after."""
    qb = qb.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    decay = 0.5 ** (1 / QB_HALFLIFE_GAMES)
    k, prior = QB_PRIOR_DROPBACKS, QB_PRIOR_EPA
    state, pre, post = {}, np.empty(len(qb)), np.empty(len(qb))
    for i, (qid, db, e) in enumerate(zip(qb["qb_id"], qb["db"], qb["epa_sum"])):
        E, D = state.get(qid, (0.0, 0.0))
        pre[i] = (E + k * prior) / (D + k)
        E, D = decay * E + e, decay * D + db
        state[qid] = (E, D)
        post[i] = (E + k * prior) / (D + k)
    return qb.assign(qb_pre=pre, qb_post=post)


# --- step 4: Elo ---------------------------------------------------------------
def _elo(games: pd.DataFrame, k=20.0, hfa=48.0, revert=1 / 3, base=1500.0):
    rating, last_season, rows = {}, {}, []
    for r in games.sort_values(["gameday", "game_id"]).itertuples():
        for t in (r.home_team, r.away_team):
            if last_season.get(t) != r.season:   # new season: regress a third to the mean
                rating[t] = base + (rating.get(t, base) - base) * (1 - revert)
                last_season[t] = r.season
        h, a = rating[r.home_team], rating[r.away_team]
        rows.append((r.game_id, h, a))
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            edge = h - a + (0 if r.location == "Neutral" else hfa)
            exp_home = 1 / (1 + 10 ** (-edge / 400))
            mov = r.home_score - r.away_score
            actual = 0.5 if mov == 0 else float(mov > 0)
            winner_edge = edge if mov > 0 else -edge
            mult = np.log(abs(mov) + 1) * 2.2 / (winner_edge * 0.001 + 2.2)
            delta = k * mult * (actual - exp_home)
            rating[r.home_team] += delta
            rating[r.away_team] -= delta
    return pd.DataFrame(rows, columns=["game_id", "h_elo", "a_elo"]), rating


# --- step 5: assemble ---------------------------------------------------------
def build_features(games: pd.DataFrame, pbp: pd.DataFrame) -> FeatureSet:
    tg = _team_game_table(games, pbp)
    pre, latest = _rolling_states(tg)

    # quarterbacks: starter = QB with the most dropbacks in that team-game
    qb = _qb_ratings(_qb_game_stats(pbp).merge(games[["game_id", "gameday"]], on="game_id"))
    starters = qb.sort_values("db").groupby(["game_id", "team"]).tail(1)
    starters = starters[["game_id", "team", "qb_id", "qb_name", "qb_pre", "qb_post"]]
    # raw per-game stats share names with the pre-game states, so drop them first
    tg = tg.drop(columns=EWMA_COLS).join(pre).merge(starters, on=["game_id", "team"], how="left")

    # upcoming games: each team's most recent starter, at his latest rating
    last_starter = (tg[tg["played"]].sort_values("gameday").groupby("team").tail(1)
                    .set_index("team")[["qb_id", "qb_name", "qb_post"]])
    fut = ~tg["played"]
    tg.loc[fut, "qb_id"] = tg.loc[fut, "team"].map(last_starter["qb_id"])
    tg.loc[fut, "qb_name"] = tg.loc[fut, "team"].map(last_starter["qb_name"])
    tg["qb_epa"] = np.where(fut, tg["team"].map(last_starter["qb_post"]), tg["qb_pre"])

    elo_df, elo_now = _elo(games)

    keep = ["game_id", "team", "played", "qb_id", "qb_name", "qb_epa"] + EWMA_COLS + ["form_net_epa"]
    side_cols = [c for c in keep if c not in ("game_id", "team")]

    gf = games[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
                "home_score", "away_score", "result", "location", "home_rest", "away_rest",
                "spread_line", "div_game"]].copy()
    for pfx, team_col in (("h_", "home_team"), ("a_", "away_team")):
        s = tg[keep].rename(columns={c: pfx + c for c in side_cols})
        gf = gf.merge(s.rename(columns={"team": team_col}), on=["game_id", team_col], how="left")
    gf = gf.merge(elo_df, on="game_id", how="left")

    for c in STATE_COLS:
        gf[f"d_{c}"] = gf[f"h_{c}"] - gf[f"a_{c}"]
    gf["rest_diff"] = (gf["home_rest"].fillna(7) - gf["away_rest"].fillna(7)).clip(-7, 7)
    gf["home_win"] = np.where(gf["result"].isna() | (gf["result"] == 0), np.nan, (gf["result"] > 0).astype(float))

    # current state per team / per QB, for ad-hoc "what if" matchups
    lt = latest.copy()
    lt["elo"] = pd.Series(elo_now)
    lt["qb_epa"] = last_starter["qb_post"]
    lt["qb_name"] = last_starter["qb_name"]
    lt["qb_id"] = last_starter["qb_id"]
    qbs = (qb.sort_values("gameday").groupby("qb_id").tail(1)
           [["qb_id", "qb_name", "team", "gameday", "qb_post"]].rename(columns={"qb_post": "qb_epa"}))
    return FeatureSet(gf, lt, qbs.reset_index(drop=True))


def matchup_row(fs: FeatureSet, home: str, away: str,
                home_qb: pd.Series | None = None, away_qb: pd.Series | None = None,
                rest_diff: float = 0.0) -> pd.DataFrame:
    """Build a one-row feature frame for any hypothetical matchup, using each
    team's latest state. Optionally swap in a different starting QB."""
    L = fs.latest
    for t in (home, away):
        if t not in L.index:
            raise SystemExit(f"Unknown team '{t}'. Known: {', '.join(sorted(L.index))}")
    h, a = L.loc[home].copy(), L.loc[away].copy()
    if home_qb is not None:
        h["qb_epa"], h["qb_name"] = home_qb["qb_epa"], home_qb["qb_name"]
    if away_qb is not None:
        a["qb_epa"], a["qb_name"] = away_qb["qb_epa"], away_qb["qb_name"]
    row = {f"h_{c}": h[c] for c in STATE_COLS + ["qb_name"]}
    row.update({f"a_{c}": a[c] for c in STATE_COLS + ["qb_name"]})
    row.update({f"d_{c}": float(h[c]) - float(a[c]) for c in STATE_COLS})
    row["rest_diff"] = rest_diff
    return pd.DataFrame([row])


# --- persistence ---------------------------------------------------------------
def save_features(fs: FeatureSet) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    fs.games.to_csv(CACHE / "features_games.csv", index=False)
    fs.latest.to_csv(CACHE / "features_latest.csv")
    fs.qbs.to_csv(CACHE / "features_qbs.csv", index=False)


def load_features() -> FeatureSet:
    try:
        games = pd.read_csv(CACHE / "features_games.csv", parse_dates=["gameday"])
        latest = pd.read_csv(CACHE / "features_latest.csv", index_col=0)
        qbs = pd.read_csv(CACHE / "features_qbs.csv", parse_dates=["gameday"])
    except FileNotFoundError:
        raise SystemExit("No features found. Run:  python run.py update")
    return FeatureSet(games, latest, qbs)
