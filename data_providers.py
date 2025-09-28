import pandas as pd
import numpy as np

# New: use nfl_data_py instead of hand-written GitHub URLs
import nfl_data_py as nfl

# Shown in the Streamlit header so you can confirm the running build
CODE_VERSION = "v2.0-nfl_data_py"

# If a brand-new season isn't fully published yet in nflverse,
# we can optionally fall back to last season to keep the app alive.
FALLBACK_SEASON = None  # set e.g. 2024 if you want a forced fallback


# ============ SCHEDULES ============
def fetch_schedule() -> pd.DataFrame:
    """
    Return all schedules available (all seasons) with the columns we use.
    nfl_data_py handles fetching from the correct nflverse-data releases.
    """
    # nfl.import_schedules returns a big table across seasons
    sched = nfl.import_schedules(years=True)  # True = all available seasons
    # Normalize to the columns your app expects
    cols = ["season", "week", "gameday", "away_team", "home_team", "game_type"]
    out = pd.DataFrame({
        "season": sched["season"],
        "week": sched["week"],
        "gameday": sched["gameday"],
        "away_team": sched["away_team"].str.upper(),
        "home_team": sched["home_team"].str.upper(),
        "game_type": sched["game_type"].str.upper(),
    })
    # Filter to only real NFL games (REG/POST); keep PRE if you want preseason picks
    out = out[out["game_type"].isin(["REG", "POST"])]
    return out[cols]


# ============ PLAY-BY-PLAY ============
def fetch_pbp_season(season: int) -> pd.DataFrame:
    """
    Load play-by-play for a given season using nfl_data_py.
    If the requested season isn't fully available (early in the year), optionally
    fall back to a previous season to keep the app running.
    """
    season = int(season)
    seasons = [season]
    fb_used = ""

    try:
        pbp = nfl.import_pbp_data(seasons)
        if pbp.empty and FALLBACK_SEASON:
            fb_used = f"⚠️ {season} PBP not fully available — using {FALLBACK_SEASON}"
            pbp = nfl.import_pbp_data([FALLBACK_SEASON]).copy()
            # Keep downstream filters working as if it's the requested season
            if "season" in pbp.columns:
                pbp["season"] = season
    except Exception:
        if FALLBACK_SEASON:
            fb_used = f"⚠️ {season} PBP load error — using {FALLBACK_SEASON}"
            pbp = nfl.import_pbp_data([FALLBACK_SEASON]).copy()
            if "season" in pbp.columns:
                pbp["season"] = season
        else:
            # Surface a clean empty DF; the UI will show "insufficient data"
            pbp = pd.DataFrame()

    if not pbp.empty and "__fallback_notice__" not in pbp.columns:
        pbp["__fallback_notice__"] = fb_used
    return pbp


# ============ METRICS BUILDERS ============
def compute_team_unit_metrics(pbp: pd.DataFrame, season: int, week: int):
    """
    Build offense and defense unit tables from PBP up to selected week (REG).
    Offense: EPA/play, success%, explosive% + OL proxies (pass/run block win).
    Defense: EPA/success/explosive allowed + pressure and run-stop proxies.
    """
    if pbp is None or pbp.empty:
        return (
            pd.DataFrame(columns=["team", "unit", "epa_per_play", "success_rate",
                                  "explosive_rate", "pass_block_win", "run_block_win"]),
            pd.DataFrame(columns=["team", "unit", "epa_allowed", "success_allowed",
                                  "explosive_allowed", "pressure_rate",
                                  "run_stop_win", "coverage_grade"])
        )

    df = pbp.copy()
    # nfl_data_py already includes season_type column (REG/POST)
    if "season_type" in df.columns:
        df = df[(df["season"] == season) &
                (df["week"] <= int(week)) &
                (df["season_type"].str.upper() == "REG")]
    else:
        df = df[(df["season"] == season) & (df["week"] <= int(week))]

    # Keep only run/pass plays with valid EPA
    df = df[df["play_type"].isin(["pass", "run"]) & df["epa"].notna()].copy()

    # Success/explosive definitions
    df["success"] = (df["epa"] > 0).astype(int)
    air = df["air_yards"] if "air_yards" in df.columns else np.nan
    df["explosive"] = np.where(
        (df["play_type"] == "pass") & (air >= 20), 1,
        np.where((df["play_type"] == "run") & (df["yards_gained"] >= 12), 1, 0)
    )

    # ----- Offense aggregates -----
    off = (df.groupby("posteam", dropna=True)
             .agg(epa_per_play=("epa", "mean"),
                  success_rate=("success", "mean"),
                  explosive_rate=("explosive", "mean"))
             .reset_index().rename(columns={"posteam": "team"}))

    # OL proxies
    pass_df = df[df["play_type"] == "pass"].copy()
    if "sack" not in pass_df.columns:
        pass_df["sack"] = 0
    pass_df["sack"] = pass_df["sack"].fillna(0).astype(int)
    ol_pass = (pass_df.groupby("posteam", dropna=True)
                    .agg(attempts=("play_type", "count"),
                         sacks=("sack", "sum"))
                    .reset_index().rename(columns={"posteam": "team"}))
    ol_pass["pass_block_win_proxy"] = 1.0 - (ol_pass["sacks"] / ol_pass["attempts"].clip(lower=1))

    run_df = df[df["play_type"] == "run"].copy()
    run_df["stuffed"] = (run_df["yards_gained"] <= 0).astype(int)
    ol_run = (run_df.groupby("posteam", dropna=True)
                    .agg(runs=("play_type", "count"),
                         stuffed=("stuffed", "sum"))
                    .reset_index().rename(columns={"posteam": "team"}))
    ol_run["run_block_win_proxy"] = 1.0 - (ol_run["stuffed"] / ol_run["runs"].clip(lower=1))

    ol = pd.merge(
        ol_pass[["team", "pass_block_win_proxy"]],
        ol_run[["team", "run_block_win_proxy"]],
        on="team", how="outer",
    )

    # Offense units table
    offense_units = []
    for _, r in off.iterrows():
        t = r["team"]
        offense_units += [
            {"team": t, "unit": "QB", "epa_per_play": r["epa_per_play"], "success_rate": r["success_rate"],
             "explosive_rate": r["explosive_rate"], "pass_block_win": np.nan, "run_block_win": np.nan},
            {"team": t, "unit": "RB", "epa_per_play": r["epa_per_play"], "success_rate": r["success_rate"],
             "explosive_rate": r["explosive_rate"], "pass_block_win": np.nan, "run_block_win": np.nan},
            {"team": t, "unit": "WR", "epa_per_play": r["epa_per_play"], "success_rate": r["success_rate"],
             "explosive_rate": r["explosive_rate"], "pass_block_win": np.nan, "run_block_win": np.nan},
            {"team": t, "unit": "TE", "epa_per_play": r["epa_per_play"], "success_rate": r["success_rate"],
             "explosive_rate": r["explosive_rate"], "pass_block_win": np.nan, "run_block_win": np.nan},
        ]
    for _, r in ol.iterrows():
        offense_units.append({
            "team": r["team"], "unit": "OL",
            "epa_per_play": np.nan, "success_rate": np.nan, "explosive_rate": np.nan,
            "pass_block_win": r["pass_block_win_proxy"], "run_block_win": r["run_block_win_proxy"]
        })
    offense_units = pd.DataFrame(offense_units)

    # ----- Defense aggregates -----
    deff = (df.groupby("defteam", dropna=True)
              .agg(epa_allowed=("epa", "mean"),
                   success_allowed=("success", "mean"),
                   explosive_allowed=("explosive", "mean"))
              .reset_index().rename(columns={"defteam": "team"}))

    def_pass = (pass_df.groupby("defteam", dropna=True)
                  .agg(attempts=("play_type", "count"),
                       sacks=("sack", "sum"))
                  .reset_index().rename(columns={"defteam": "team"}))
    def_pass["pressure_rate_proxy"] = (def_pass["sacks"] / def_pass["attempts"].clip(lower=1))

    def_run = (run_df.groupby("defteam", dropna=True)
                 .agg(runs=("play_type", "count"),
                      stuffs=("stuffed", "sum"))
                 .reset_index().rename(columns={"defteam": "team"}))
    def_run["run_stop_win_proxy"] = (def_run["stuffs"] / def_run["runs"].clip(lower=1))

    deff = (
        deff.merge(def_pass[["team", "pressure_rate_proxy"]], on="team", how="left")
            .merge(def_run[["team", "run_stop_win_proxy"]], on="team", how="left")
    )
    deff["coverage_grade"] = np.nan  # placeholder for optional enrichment

    defense_units = []
    for _, r in deff.iterrows():
        t = r["team"]
        defense_units += [
            {"team": t, "unit": "PassRush", "epa_allowed": np.nan, "success_allowed": np.nan,
             "explosive_allowed": np.nan, "pressure_rate": r.get("pressure_rate_proxy", np.nan),
             "run_stop_win": np.nan, "coverage_grade": np.nan},
            {"team": t, "unit": "RunDefense", "epa_allowed": np.nan, "success_allowed": np.nan,
             "explosive_allowed": r["explosive_allowed"], "pressure_rate": np.nan,
             "run_stop_win": r.get("run_stop_win_proxy", np.nan), "coverage_grade": np.nan},
            {"team": t, "unit": "CoverageDB", "epa_allowed": r["epa_allowed"], "success_allowed": r["success_allowed"],
             "explosive_allowed": r["explosive_allowed"], "pressure_rate": np.nan,
             "run_stop_win": np.nan, "coverage_grade": np.nan},
            {"team": t, "unit": "CoverageLB", "epa_allowed": r["epa_allowed"], "success_allowed": r["success_allowed"],
             "explosive_allowed": r["explosive_allowed"], "pressure_rate": np.nan,
             "run_stop_win": np.nan, "coverage_grade": np.nan},
            {"team": t, "unit": "DL", "epa_allowed": np.nan, "success_allowed": np.nan,
             "explosive_allowed": np.nan, "pressure_rate": np.nan,
             "run_stop_win": r.get("run_stop_win_proxy", np.nan), "coverage_grade": np.nan},
        ]
    defense_units = pd.DataFrame(defense_units)

    return offense_units, defense_units


# ============ OPTIONAL ENRICHMENT STUBS ============
def enrich_with_espn_winrates(offense_units: pd.DataFrame, defense_units: pd.DataFrame, enabled: bool):
    """Placeholder for ESPN pass/rush block win-rate ingestion."""
    return offense_units, defense_units, False


def enrich_with_sportsdataio(offense_units: pd.DataFrame, defense_units: pd.DataFrame, api_key: str | None):
    """Placeholder for SportsDataIO-based enrichment."""
    return offense_units, defense_units, False
