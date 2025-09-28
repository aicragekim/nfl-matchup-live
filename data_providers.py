import io
import gzip
import requests
import pandas as pd
import numpy as np

# Shown in the Streamlit header so you can confirm the running build
CODE_VERSION = "v1.3-sched-espn-pbp-fallback"

# ---------- SCHEDULE SOURCES ----------
# nflverse (try .gz then .csv)
SCHEDULE_URLS = [
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules.csv.gz",
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules.csv",
]

# ESPN scoreboard (used as season-specific fallback)
# Example: https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard?week=3&seasontype=2&dates=2025

# ---------- PLAY-BY-PLAY SOURCE ----------
# Published per season; some future seasons may not exist yet
PBP_PARQUET_TMPL = (
    "https://github.com/nflverse/nflfastR-data/releases/download/pbp_{season}/"
    "play_by_play_{season}.parquet"
)
PBP_FALLBACK_SEASON = 2024  # latest widely available parquet at time of writing


# ============ HTTP UTIL ============
def _http_get(url, headers=None, timeout=30):
    h = {"User-Agent": "matchup-app/1.0"}
    if headers:
        h.update(headers)
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


# ============ SCHEDULES ============
def _fetch_schedule_nflverse() -> pd.DataFrame:
    """
    Try nflverse schedules from known URLs. Returns all seasons if available.
    """
    last_err = None
    for url in SCHEDULE_URLS:
        try:
            r = _http_get(url)
            if url.endswith(".gz"):
                df = pd.read_csv(io.BytesIO(gzip.decompress(r.content)))
            else:
                df = pd.read_csv(io.BytesIO(r.content))
            cols = ["season", "week", "gameday", "away_team", "home_team", "game_type"]
            if not set(cols).issubset(df.columns):
                raise ValueError(f"Missing columns: {set(cols)-set(df.columns)}")
            return df[cols]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"nflverse schedules unavailable. Last error: {last_err}")


def _fetch_schedule_espn_full_season(season: int) -> pd.DataFrame:
    """
    Build a REG-season schedule by pulling ESPN's scoreboard for weeks 1..18.
    """
    rows = []
    for wk in range(1, 19):
        url = f"https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard?week={wk}&seasontype=2&dates={season}"
        try:
            data = _http_get(url).json()
        except Exception:
            continue  # skip week if ESPN hiccups
        for ev in data.get("events", []):
            comps = (ev.get("competitions") or [{}])[0]
            teams = comps.get("competitors") or []
            if len(teams) != 2:
                continue
            home = next((t for t in teams if t.get("homeAway") == "home"), None)
            away = next((t for t in teams if t.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            home_abbr = (((home.get("team") or {}).get("abbreviation")) or "").upper()
            away_abbr = (((away.get("team") or {}).get("abbreviation")) or "").upper()
            gameday = ev.get("date")  # ISO timestamp string
            if home_abbr and away_abbr:
                rows.append({
                    "season": season,
                    "week": wk,
                    "gameday": gameday,
                    "away_team": away_abbr,
                    "home_team": home_abbr,
                    "game_type": "REG",
                })
    return pd.DataFrame(rows, columns=["season", "week", "gameday", "away_team", "home_team", "game_type"])


def fetch_schedule() -> pd.DataFrame:
    """
    Try nflverse first (all seasons). If it fails, return an empty shell.
    streamlit_app will detect empty and call the ESPN season builder.
    """
    try:
        return _fetch_schedule_nflverse()
    except Exception:
        return pd.DataFrame(columns=["season", "week", "gameday", "away_team", "home_team", "game_type"])


def fetch_schedule_for_season_from_espn(season: int) -> pd.DataFrame:
    """Build a REG-season schedule for a single season via ESPN."""
    return _fetch_schedule_espn_full_season(season)


# ============ PLAY-BY-PLAY ============
def fetch_pbp_season(season: int) -> pd.DataFrame:
    """
    Try to fetch play-by-play parquet for the requested season.
    If not available (e.g., future season), fall back to PBP_FALLBACK_SEASON.
    Adds a __fallback_notice__ column if a fallback was used.
    """
    season = int(season)
    url = PBP_PARQUET_TMPL.format(season=season)
    try:
        r = _http_get(url)
        df = pd.read_parquet(io.BytesIO(r.content), engine="pyarrow")
        df["__fallback_notice__"] = ""
        return df
    except Exception:
        fb = int(PBP_FALLBACK_SEASON)
        fb_url = PBP_PARQUET_TMPL.format(season=fb)
        r = _http_get(fb_url)
        df = pd.read_parquet(io.BytesIO(r.content), engine="pyarrow")
        df["__fallback_notice__"] = f"⚠️ {season} PBP not yet available — using {fb}"
        # Normalize season field so downstream filters still work using caller's season
        if "season" in df.columns:
            df["season"] = season
        return df


# ============ METRICS BUILDERS ============
def compute_team_unit_metrics(pbp: pd.DataFrame, season: int, week: int):
    """
    Build offense and defense unit tables from PBP up to selected week (REG).
    Offense: EPA/play, success%, explosive% + OL proxies (pass/run block win).
    Defense: EPA/success/explosive allowed + pressure and run-stop proxies.
    """
    df = pbp.copy()
    df = df[(df["season"] == season) & (df["week"] <= int(week)) & (df["season_type"] == "REG")]
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
    """
    Placeholder for ESPN pass/rush block win-rate ingestion.
    """
    return offense_units, defense_units, False


def enrich_with_sportsdataio(offense_units: pd.DataFrame, defense_units: pd.DataFrame, api_key: str | None):
    """
    Placeholder for SportsDataIO-based enrichment.
    """
    return offense_units, defense_units, False
