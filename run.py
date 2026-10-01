#!/usr/bin/env python3
"""NFL win-probability model.

  python run.py update                      download data + build features
  python run.py backtest                    walk-forward evaluation (no peeking)
  python run.py train                       fit models on all completed games
  python run.py week                        probabilities + drivers for the upcoming week
  python run.py predict --home BUF --away KC   any matchup, with a full breakdown
"""
import argparse

import pandas as pd

from nflwp.config import DEFAULT_MODEL, FIRST_SEASON, MODEL_DIR
from nflwp.data import load_games, load_pbp
from nflwp.features import FEATURES, build_features, load_features, matchup_row, save_features
from nflwp.model import (MODEL_NAMES, calibration_table, explain, global_importance,
                         load_model, predict_home_prob, score_table, train_and_save,
                         walk_forward)

pd.set_option("display.width", 200)


# ---------------------------------------------------------------- commands
def cmd_update(args):
    games = load_games(refresh=True)
    last = int(games["season"].max())
    current = int(games.loc[games["result"].notna(), "season"].max())
    pbp = load_pbp(FIRST_SEASON, last, refresh_season=current)
    fs = build_features(games, pbp)
    save_features(fs)
    g = fs.games
    print(f"Built features: {int(g['home_win'].notna().sum())} completed games, "
          f"{int(g['result'].isna().sum())} upcoming. Latest results: season {current}.")


def cmd_backtest(args):
    fs = load_features()
    print(f"Walk-forward backtest {args.first}-{args.last} (each season predicted by models trained on earlier seasons)")
    bt = walk_forward(fs.games, args.first, args.last)
    MODEL_DIR.mkdir(exist_ok=True)
    bt.to_csv(MODEL_DIR / "backtest_predictions.csv", index=False)

    print("\n=== Out-of-sample scores (lower log_loss / brier is better) ===")
    print(score_table(bt).round(4).to_string())
    print("\nReference points: always_home = predict the base rate; elo_only = a simple Elo model;")
    print("vegas = closing point spread converted to a probability (the bar to beat).")

    print(f"\n=== Calibration of '{args.model}' (does 70% mean 70%?) ===")
    print(calibration_table(bt["y"], bt[args.model]).round(3).to_string())

    print(f"\n=== What matters? (holdout season {args.last}) ===")
    print(global_importance(fs.games, args.last).round(4).to_string())
    print("\nperm_importance_logloss: how much worse the boosted model gets when the feature is shuffled.")
    print("logistic_std_coef: effect per 1 std-dev of the feature (correlated features can offset each other).")


def cmd_train(args):
    train_and_save(load_features().games)
    print(f"Saved models to {MODEL_DIR}/")


def cmd_week(args):
    fs, model = load_features(), load_model(args.model)
    g = fs.games
    if args.season is None:
        first = g[g["result"].isna()].sort_values("gameday").iloc[0]
        args.season, args.week = int(first["season"]), int(first["week"])
    wk = g[(g["season"] == args.season) & (g["week"] == args.week)].sort_values(["gameday", "game_id"])
    if wk.empty:
        raise SystemExit("No games found for that season/week.")

    print(f"\nSeason {args.season}, week {args.week}  |  model: {args.model}\n")
    out = []
    for _, r in wk.iterrows():
        neutral = r["location"] == "Neutral"
        x = pd.DataFrame([r[FEATURES].astype(float)])
        p = float(predict_home_prob(model, x, neutral)[0])
        _, phi = explain(model, r[FEATURES], neutral)
        top = sorted(phi.items(), key=lambda kv: -abs(kv[1]))[:3]
        drivers = ", ".join(f"{k} {_side(v, r.home_team, r.away_team)} {abs(v)*100:.1f}" for k, v in top)
        result = ""
        if pd.notna(r["result"]):
            result = f"  FINAL {r.away_team} {int(r.away_score)}-{int(r.home_score)} {r.home_team}"
        fav, fp = (r.home_team, p) if p >= 0.5 else (r.away_team, 1 - p)
        print(f"{r['gameday']:%a %m-%d}  {r.away_team:>3} @ {r.home_team:<3}  "
              f"{r.home_team} {p*100:4.1f}% | {r.away_team} {(1-p)*100:4.1f}%   "
              f"pick: {fav:<3}  drivers: {drivers}{result}")
        out.append({"game_id": r["game_id"], "gameday": r["gameday"], "home": r.home_team,
                    "away": r.away_team, "home_win_prob": p, **{f"pp_{k}": v * 100 for k, v in phi.items()}})
    if args.csv:
        pd.DataFrame(out).to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")
    if wk["result"].notna().any():
        print("\nGames marked FINAL were part of the training data, so those picks are in-sample.")
    print("\n(drivers = biggest factors, in percentage points of win probability, and which team they favor)")
    print("Note: predictions use each team's rating going into the game; injuries/QB changes aren't known yet.")


def cmd_predict(args):
    fs, model = load_features(), load_model(args.model)
    home, away = args.home.upper(), args.away.upper()
    hq = _find_qb(fs, args.home_qb) if args.home_qb else None
    aq = _find_qb(fs, args.away_qb) if args.away_qb else None
    row = matchup_row(fs, home, away, hq, aq)
    row["rest_diff"] = float(args.rest_diff)
    r = row.iloc[0]

    p = float(predict_home_prob(model, row, args.neutral)[0])
    _, phi = explain(model, r[FEATURES], args.neutral)

    site = "neutral site" if args.neutral else f"at {home}"
    print(f"\n{away} @ {home}  ({site})   model: {args.model}\n")
    print(f"  {home:<4} {p*100:5.1f}%  {'#' * int(round(p * 40))}")
    print(f"  {away:<4} {(1-p)*100:5.1f}%  {'#' * int(round((1-p) * 40))}\n")

    print("What drove it (start at a 50% coin flip; percentage points of win probability):")
    for k, v in sorted(phi.items(), key=lambda kv: -abs(kv[1])):
        team = home if v >= 0 else away
        bar = "█" * min(20, int(round(abs(v) * 100 / 0.5)))
        print(f"  {k:<14} {v*100:+5.1f} pp  favors {team:<3} {bar}")
    print(f"  {'TOTAL':<14} {(p-0.5)*100:+5.1f} pp\n")

    print(f"Matchup snapshot ({home} vs {away}); higher is better for every row")
    rows = [
        ("Elo rating", "elo", "{:.0f}"), ("Point diff / game (recent-weighted)", "pt_diff", "{:+.1f}"),
        ("Offense EPA/play, passing", "off_epa_pass", "{:+.3f}"), ("Offense EPA/play, rushing", "off_epa_rush", "{:+.3f}"),
        ("Defense EPA/play, vs pass", "def_epa_pass", "{:+.3f}"), ("Defense EPA/play, vs rush", "def_epa_rush", "{:+.3f}"),
        ("QB EPA/dropback (shrunk)", "qb_epa", "{:+.3f}"), ("Turnover margin / game", "turnover_margin", "{:+.2f}"),
        ("Last-3-games net EPA/play", "form_net_epa", "{:+.3f}"),
    ]
    print(f"  {'':<38}{home:>11}{away:>11}")
    for label, col, f in rows:
        print(f"  {label:<38}{f.format(r['h_' + col]):>11}{f.format(r['a_' + col]):>11}")
    print(f"  {'Starting QB':<38}{str(r['h_qb_name']):>11}{str(r['a_qb_name']):>11}\n")


# ---------------------------------------------------------------- helpers
def _side(v, home, away):
    return f"{'→' + home if v >= 0 else '→' + away}"


def _find_qb(fs, query):
    m = fs.qbs[fs.qbs["qb_name"].str.contains(query, case=False, na=False)].sort_values("gameday", ascending=False)
    if m.empty:
        raise SystemExit(f"No quarterback matching '{query}' (names look like 'P.Mahomes').")
    if len(m) > 1:
        print(f"  note: '{query}' matched {len(m)} QBs; using {m.iloc[0]['qb_name']} ({m.iloc[0]['team']}).")
    return m.iloc[0]


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", help="download data and build features").set_defaults(fn=cmd_update)

    p = sub.add_parser("backtest", help="walk-forward evaluation")
    p.add_argument("--first", type=int, default=2018)
    p.add_argument("--last", type=int, default=2025)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_NAMES)
    p.set_defaults(fn=cmd_backtest)

    sub.add_parser("train", help="fit and save all models").set_defaults(fn=cmd_train)

    p = sub.add_parser("week", help="predict a week of games")
    p.add_argument("--season", type=int)
    p.add_argument("--week", type=int)
    p.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_NAMES)
    p.add_argument("--csv", help="also write results to this CSV")
    p.set_defaults(fn=cmd_week)

    p = sub.add_parser("predict", help="predict any matchup with a full breakdown")
    p.add_argument("--home", required=True, help="home team abbreviation, e.g. BUF")
    p.add_argument("--away", required=True, help="away team abbreviation, e.g. KC")
    p.add_argument("--neutral", action="store_true", help="neutral-site game (no home edge)")
    p.add_argument("--home-qb", help="what-if: swap in a different starting QB, e.g. 'Allen'")
    p.add_argument("--away-qb", help="what-if: swap in a different starting QB")
    p.add_argument("--rest-diff", type=float, default=0.0, help="home rest days minus away rest days")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_NAMES)
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
