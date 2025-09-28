
import io, gzip, requests, pandas as pd, numpy as np

NFLVERSE_SCHEDULE_GZ = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules.csv.gz"

PBP_PARQUET_TMPL = "https://github.com/nflverse/nflfastR-data/releases/download/pbp_{season}/play_by_play_{season}.parquet"

def _http_get(url, headers=None):
    r = requests.get(url, headers=headers or {"User-Agent":"matchup-app/1.0"}, timeout=30)
    r.raise_for_status()
    return r

def fetch_schedule():
    r = _http_get(NFLVERSE_SCHEDULE_GZ)
    return pd.read_csv(io.BytesIO(gzip.decompress(r.content)))[["season","week","gameday","away_team","home_team","game_type"]]

def fetch_pbp_season(season:int):
    url = PBP_PARQUET_TMPL.format(season=season)
    r = _http_get(url)
    return pd.read_parquet(io.BytesIO(r.content), engine="pyarrow")

def compute_team_unit_metrics(pbp: pd.DataFrame, season:int, week:int):
    df = pbp.copy()
    df = df[(df["season"]==season) & (df["week"]<=week) & (df["season_type"]=="REG")]
    df = df[df["play_type"].isin(["pass","run"]) & (df["epa"].notna())]
    df["success"] = (df["epa"]>0).astype(int)
    df["explosive"] = np.where((df["play_type"]=="pass") & (df["air_yards"]>=20), 1,
                         np.where((df["play_type"]=="run") & (df["yards_gained"]>=12), 1, 0))
    off = (df.groupby("posteam", dropna=True)
             .agg(epa_per_play=("epa","mean"),
                  success_rate=("success","mean"),
                  explosive_rate=("explosive","mean"))
             .reset_index().rename(columns={"posteam":"team"}))

    pass_df = df[df["play_type"]=="pass"].copy()
    pass_df["sack"] = pass_df.get("sack", 0)
    try:
        pass_df["sack"] = pass_df["sack"].fillna(0).astype(int)
    except:
        pass_df["sack"] = 0
    ol_pass = (pass_df.groupby("posteam", dropna=True)
                 .agg(attempts=("play_type","count"), sacks=("sack","sum"))
                 .reset_index().rename(columns={"posteam":"team"}))
    ol_pass["pass_block_win_proxy"] = 1.0 - (ol_pass["sacks"] / ol_pass["attempts"].clip(lower=1))

    run_df = df[df["play_type"]=="run"].copy()
    run_df["stuffed"] = (run_df["yards_gained"]<=0).astype(int)
    ol_run = (run_df.groupby("posteam", dropna=True)
                .agg(runs=("play_type","count"), stuffed=("stuffed","sum"))
                .reset_index().rename(columns={"posteam":"team"}))
    ol_run["run_block_win_proxy"] = 1.0 - (ol_run["stuffed"] / ol_run["runs"].clip(lower=1))

    ol = pd.merge(ol_pass[["team","pass_block_win_proxy"]],
                  ol_run[["team","run_block_win_proxy"]],
                  on="team", how="outer")

    offense_units = []
    for _, r in off.iterrows():
        t = r["team"]
        offense_units += [
            {"team":t,"unit":"QB","epa_per_play":r["epa_per_play"],"success_rate":r["success_rate"],"explosive_rate":r["explosive_rate"],"pass_block_win":float("nan"),"run_block_win":float("nan")},
            {"team":t,"unit":"RB","epa_per_play":r["epa_per_play"],"success_rate":r["success_rate"],"explosive_rate":r["explosive_rate"],"pass_block_win":float("nan"),"run_block_win":float("nan")},
            {"team":t,"unit":"WR","epa_per_play":r["epa_per_play"],"success_rate":r["success_rate"],"explosive_rate":r["explosive_rate"],"pass_block_win":float("nan"),"run_block_win":float("nan")},
            {"team":t,"unit":"TE","epa_per_play":r["epa_per_play"],"success_rate":r["success_rate"],"explosive_rate":r["explosive_rate"],"pass_block_win":float("nan"),"run_block_win":float("nan")},
        ]
    for _, r in ol.iterrows():
        offense_units.append({"team":r["team"],"unit":"OL","epa_per_play":float("nan"),"success_rate":float("nan"),"explosive_rate":float("nan"),
                              "pass_block_win":r["pass_block_win_proxy"],"run_block_win":r["run_block_win_proxy"]})
    offense_units = pd.DataFrame(offense_units)

    def_grp = (df.groupby("defteam", dropna=True)
                 .agg(epa_allowed=("epa","mean"),
                      success_allowed=("success","mean"),
                      explosive_allowed=("explosive","mean"))
                 .reset_index().rename(columns={"defteam":"team"}))

    def_pass = (pass_df.groupby("defteam", dropna=True)
                  .agg(attempts=("play_type","count"), sacks=("sack","sum"))
                  .reset_index().rename(columns={"defteam":"team"}))
    def_pass["pressure_rate_proxy"] = (def_pass["sacks"] / def_pass["attempts"].clip(lower=1))

    def_run = (run_df.groupby("defteam", dropna=True)
                 .agg(runs=("play_type","count"), stuffs=("stuffed","sum"))
                 .reset_index().rename(columns={"defteam":"team"}))
    def_run["run_stop_win_proxy"] = (def_run["stuffs"] / def_run["runs"].clip(lower=1))

    deff = (def_grp.merge(def_pass[["team","pressure_rate_proxy"]], on="team", how="left")
                   .merge(def_run[["team","run_stop_win_proxy"]], on="team", how="left"))
    deff["coverage_grade"] = float("nan")

    defense_units = []
    for _, r in deff.iterrows():
        t=r["team"]
        defense_units += [
            {"team":t,"unit":"PassRush","epa_allowed":float("nan"),"success_allowed":float("nan"),"explosive_allowed":float("nan"),
             "pressure_rate":r.get("pressure_rate_proxy",float("nan")),"run_stop_win":float("nan"),"coverage_grade":float("nan")},
            {"team":t,"unit":"RunDefense","epa_allowed":float("nan"),"success_allowed":float("nan"),"explosive_allowed":r["explosive_allowed"],
             "pressure_rate":float("nan"),"run_stop_win":r.get("run_stop_win_proxy",float("nan")),"coverage_grade":float("nan")},
            {"team":t,"unit":"CoverageDB","epa_allowed":r["epa_allowed"],"success_allowed":r["success_allowed"],"explosive_allowed":r["explosive_allowed"],
             "pressure_rate":float("nan"),"run_stop_win":float("nan"),"coverage_grade":float("nan")},
            {"team":t,"unit":"CoverageLB","epa_allowed":r["epa_allowed"],"success_allowed":r["success_allowed"],"explosive_allowed":r["explosive_allowed"],
             "pressure_rate":float("nan"),"run_stop_win":float("nan"),"coverage_grade":float("nan")},
            {"team":t,"unit":"DL","epa_allowed":float("nan"),"success_allowed":float("nan"),"explosive_allowed":float("nan"),
             "pressure_rate":float("nan"),"run_stop_win":r.get("run_stop_win_proxy",float("nan")),"coverage_grade":float("nan")},
        ]
    defense_units = pd.DataFrame(defense_units)
    return offense_units, defense_units

def enrich_with_espn_winrates(offense_units, defense_units, enabled: bool):
    # Placeholder stub, returns unchanged data and False to indicate no enrichment
    return offense_units, defense_units, False

def enrich_with_sportsdataio(offense_units, defense_units, api_key: str|None):
    # Placeholder stub, returns unchanged data and False to indicate no enrichment
    return offense_units, defense_units, False
