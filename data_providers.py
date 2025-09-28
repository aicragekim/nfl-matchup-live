
import io
import gzip
import requests
import pandas as pd
import numpy as np

# Primary and fallback locations for schedules
SCHEDULE_URLS = [
    # Primary (current layout)
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules.csv.gz",
    # Fallback if .gz path changes or 404s, try non-gz
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules.csv",
]

# PBP parquet by season (stable release URL pattern)
PBP_PARQUET_TMPL = (
    "https://github.com/nflverse/nflfastR-data/releases/download/pbp_{season}/"
    "play_by_play_{season}.parquet"
)

def _http_get(url, headers=None, timeout=30):
    h = {"User-Agent": "matchup-app/1.0"}
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    return r

def fetch_schedule() -> pd.DataFrame:
    """
    Try multiple schedule URLs (.gz first, then plain CSV).
    Return a DataFrame with the expected columns.
    """
    last_err = None
    for url in SCHEDULE_URLS:
        try:
            r = _http_get(url)
            if url.endswith(".gz"):
                buf = io.BytesIO(gzip.decompress(r.content))
                df = pd.read_csv(buf)
            else:
                df = pd.read_csv(io.BytesIO(r.content))
            cols = ["season", "week", "gameday", "away_team", "home_team", "game_type"]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Schedule file missing columns: {missing}")
            return df[cols]
        except Exception as e:
            last_err = e
            continue
    # If all attempts failed, surface the last error with context
    raise RuntimeError(f"Failed to fetch schedules from all known URLs. Last error: {last_err}")

def fetch_pbp_season(season: int) -> pd.DataFrame:
    url = PBP_PARQUET_TMPL.format(season=season)
    r = _http_get(url)
    return pd.read_parquet(io.BytesIO(r.content), engine="pyarrow")

def compute_team_unit_metrics(pbp: pd.DataFrame, season: int, week: int):
    """
    Build offense_units and defense_units from PBP up to selected week (REG season).
    Offense: EPA/play, success%, explosive%; OL proxies: sack rate allowed, run stuff rate allowed.
    Defense: EPA/success/explosive allowed; pressure proxy (sack rate), run-stop proxy (stuff rate).
    """
    df = pbp.copy()
    df = df[(df["season"] == season) & (df["week"] <= int(week)) & (df["season_type"] == "REG")]
    df = df[df["play_type"].isin(["pass", "run"]) & df["epa"].notna()].copy()

    # Success & explosive definitions
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
                    .agg(attempts=("play_type", "count"), sacks=("sack", "sum"))
                    .reset_index().rename(columns={"posteam": "team"}))
    ol_pass["pass_block_win_proxy"] = 1.0 - (ol_pass["sacks"] / ol_pass["attempts"].clip(lower=1))

    run_df = df[df["play_type"] == "run"].copy()
    run_df["stuffed"] = (run_df["yards_gained"] <= 0).astype(int)
    ol_run = (run_df.groupby("posteam", dropna=True)
                    .agg(runs=("play_type", "count"), stuffed=("stuffed", "sum"))
                    .reset_index().rename(columns={"posteam": "team"}))
    ol_run["run_block_win_proxy"] = 1.0 - (ol_run["stuffed"] / ol_run["runs"].clip(lower=1))

    ol = pd.merge(ol_pass[["team", "pass_block_win_proxy"]],
                  ol_run[["team", "run_block_win_proxy"]],
                  on="team", how="outer")

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
            "pass_block_win": r["pass_block_win_proxy"], "run_block_win": r["run_block_win_proxy"],
        })
    offense_units = pd.DataFrame(offense_units)

    # ----- Defense aggregates -----
    deff = (df.groupby("defteam", dropna=True)
              .agg(epa_allowed=("epa", "mean"),
                   success_allowed=("success", "mean"),
                   explosive_allowed=("explosive", "mean"))
              .reset_index().rename(columns={"defteam": "team"}))

    def_pass = (pass_df.groupby("defteam", dropna=True)
                  .agg(attempts=("play_type", "count"), sacks=("sack", "sum"))
                  .reset_index().rename(columns={"defteam": "team"}))
    def_pass["pressure_rate_proxy"] = (def_pass["sacks"] / def_pass["attempts"].clip(lower=1))

    def_run = (run_df.groupby("defteam", dropna=True)
                 .agg(runs=("play_type", "count"), stuffs=("stuffed", "sum"))
                 .reset_index().rename(columns={"defteam": "team"}))
    def_run["run_stop_win_proxy"] = (def_run["stuffs"] / def_run["runs"].clip(lower=1))

    deff = (deff.merge(def_pass[["team", "pressure_rate_proxy"]], on="team", how="left")
                .merge(def_run[["team", "run_stop_win_proxy"]], on="team", how="left"))
    deff["coverage_grade"] = np.nan  # placeholder for optional adapters

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

# ---- Optional enrichment stubs (leave as-is unless you add providers) ----
def enrich_with_espn_winrates(offense_units: pd.DataFrame, defense_units: pd.DataFrame, enabled: bool):
    return offense_units, defense_units, False

def enrich_with_sportsdataio(offense_units: pd.DataFrame, defense_units: pd.DataFrame, api_key: str | None):
    return offense_units, defense_units, False
