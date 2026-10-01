"""Download and cache nflverse data.

* games.csv  -> schedule, scores, rest days, roof, etc. (includes future games)
* play-by-play -> one big file per season; we keep only the columns we need
  and store a slim copy so re-runs are fast and the cache stays small.
"""
import shutil
import urllib.error
import urllib.request

import pandas as pd

from .config import CACHE, TEAM_FIX

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "play_type",
    "epa", "wp", "pass", "rush", "qb_dropback", "sack", "interception",
    "fumble_lost", "fumbled_1_team", "two_point_attempt",
    "passer_player_id", "passer_player_name",
]


def _download(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-win-prob/0.1"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def load_games(refresh: bool = False) -> pd.DataFrame:
    path = CACHE / "games.csv"
    if refresh or not path.exists():
        print("  downloading games.csv ...")
        _download(GAMES_URL, path)
    g = pd.read_csv(path)
    for col in ("home_team", "away_team"):
        g[col] = g[col].replace(TEAM_FIX)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def load_pbp_season(season: int, refresh: bool = False) -> pd.DataFrame:
    slim = CACHE / f"pbp_{season}.csv.gz"
    if refresh or not slim.exists():
        print(f"  downloading {season} play-by-play ...")
        raw = CACHE / f"_raw_{season}.csv.gz"
        _download(PBP_URL.format(season=season), raw)
        df = pd.read_csv(raw, usecols=PBP_COLS, low_memory=False)
        df.to_csv(slim, index=False)
        raw.unlink()
    return pd.read_csv(slim, low_memory=False)


def load_pbp(first: int, last: int, refresh_season: int | None = None) -> pd.DataFrame:
    """Load play-by-play for first..last. `refresh_season` is re-downloaded
    (use it for the in-progress season so new weeks get picked up)."""
    frames = []
    for season in range(first, last + 1):
        try:
            frames.append(load_pbp_season(season, refresh=(season == refresh_season)))
        except urllib.error.HTTPError as e:
            print(f"  {season}: play-by-play not available yet (HTTP {e.code})")
    pbp = pd.concat(frames, ignore_index=True)
    for col in ("posteam", "defteam"):
        pbp[col] = pbp[col].replace(TEAM_FIX)
    return pbp
