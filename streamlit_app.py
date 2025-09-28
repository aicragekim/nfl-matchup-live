
import streamlit as st
import pandas as pd
import numpy as np
from data_providers import fetch_schedule, fetch_pbp_season, compute_team_unit_metrics, enrich_with_espn_winrates, enrich_with_sportsdataio

st.set_page_config(page_title="NFL Matchup Picks — Live", layout="wide")
st.title("🏈 NFL Matchup Picks — Live (nflverse)")

with st.sidebar:
    st.header("Data source & season")
    season = st.number_input("Season (year)", min_value=2015, max_value=2100, value=2025, step=1)
    week = st.number_input("Week (1-18)", min_value=1, max_value=18, value=3, step=1)

    st.header("Optional adapters")
    espn_enable = st.toggle("Try ESPN Win Rates (experimental)", value=False)
    sdio_enable = st.toggle("Use SportsDataIO (requires key in st.secrets['SPORTSDATAIO_KEY'])", value=False)

    st.header("Weights & dependency")
    w_qb = st.slider("QB weight", 0.0, 3.0, 1.2, 0.05)
    w_rb = st.slider("RB weight", 0.0, 3.0, 0.7, 0.05)
    w_wr = st.slider("WR weight", 0.0, 3.0, 1.1, 0.05)
    w_te = st.slider("TE weight", 0.0, 3.0, 0.6, 0.05)
    w_ol = st.slider("OL weight", 0.0, 3.0, 1.1, 0.05)

    qb_cov_w = st.slider("QB vs Coverage share", 0.0, 1.0, 0.6, 0.05)
    rb_run_w = st.slider("RB vs Run D share", 0.0, 1.0, 0.65, 0.05)
    te_covlb_w = st.slider("TE vs Coverage LB share", 0.0, 1.0, 0.55, 0.05)
    ol_pass_w = st.slider("OL Pass Pro share", 0.0, 1.0, 0.6, 0.05)

    dep_strength = st.slider("Trench impact on pass game", 0.0, 2.0, 1.0, 0.05)
    close_margin = st.slider("Close-game margin (net edge)", 0.0, 1.0, 0.15, 0.01)

@st.cache_data(show_spinner=True)
def load_schedule():
    return fetch_schedule()

@st.cache_data(show_spinner=True)
def load_pbp(season:int):
    return fetch_pbp_season(season)

@st.cache_data(show_spinner=True)
def build_units(pbp, season, week, espn_enable, sdio_enable):
    off, deff = compute_team_unit_metrics(pbp, season, int(week))
    if espn_enable:
        off, deff, ok = enrich_with_espn_winrates(off, deff, True)
        if not ok:
            st.info("ESPN win-rate enrichment not implemented in this sample.")
    if sdio_enable:
        key = st.secrets.get("SPORTSDATAIO_KEY", None)
        off, deff, ok = enrich_with_sportsdataio(off, deff, key)
        if not ok and key:
            st.info("SportsDataIO enrichment stub not implemented in this sample.")
    return off, deff

def normalize(series, invert=False):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().nunique() <= 1:
        out = pd.Series(0.5, index=s.index)
    else:
        mn, mx = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            out = pd.Series(0.5, index=s.index)
        else:
            out = (s - mn) / (mx - mn)
    return 1.0 - out if invert else out

def unitize_offense(off):
    of = off.copy()
    of["epa_n"]  = normalize(of["epa_per_play"])
    of["succ_n"] = normalize(of["success_rate"])
    of["expl_n"] = normalize(of["explosive_rate"])
    of["pbw_n"]  = normalize(of["pass_block_win"])
    of["rbw_n"]  = normalize(of["run_block_win"])
    def offense_unit_score(r):
        vals = [r["pbw_n"], r["rbw_n"]] if r["unit"]=="OL" else [r["epa_n"], r["succ_n"], r["expl_n"]]
        vals = [v for v in vals if pd.notna(v)]
        return float(np.mean(vals)) if vals else np.nan
    of["unit_off_score"] = of.apply(offense_unit_score, axis=1)
    return of

def unitize_defense(deff):
    df = deff.copy()
    df["epa_allowed_n"]  = normalize(df["epa_allowed"], invert=True)
    df["succ_allowed_n"] = normalize(df["success_allowed"], invert=True)
    df["expl_allowed_n"] = normalize(df["explosive_allowed"], invert=True)
    df["pressure_n"]     = normalize(df["pressure_rate"])
    df["run_stop_win_n"] = normalize(df["run_stop_win"])
    df["coverage_n"]     = normalize(df["coverage_grade"])
    def defense_unit_score(r):
        if r["unit"]=="PassRush": vals=[r["pressure_n"]]
        elif r["unit"]=="RunDefense": vals=[r["run_stop_win_n"], r["expl_allowed_n"], r["succ_allowed_n"]]
        elif r["unit"] in ("CoverageDB","CoverageLB"): vals=[r["coverage_n"], r["epa_allowed_n"], r["succ_allowed_n"], r["expl_allowed_n"]]
        elif r["unit"]=="DL": vals=[r["run_stop_win_n"]]
        else: vals=[]
        vals=[v for v in vals if pd.notna(v)]
        return float(np.mean(vals)) if vals else np.nan
    df["unit_def_score"] = df.apply(defense_unit_score, axis=1)
    return df

def build_maps(of, df):
    off_map = {(r.team, r.unit): r.unit_off_score for r in of.itertuples()}
    def_map = {(r.team, r.unit): r.unit_def_score for r in df.itertuples()}
    return off_map, def_map

def unit_matchup(off_team, def_team, off_map, def_map, qb_cov_w, rb_run_w, te_covlb_w, ol_pass_w):
    qb_cov = def_map.get((def_team, "CoverageDB"), np.nan)
    qb_pr  = def_map.get((def_team, "PassRush"), np.nan)
    qb_off = off_map.get((off_team, "QB"), np.nan)
    qb_edge = qb_off - (qb_cov * qb_cov_w if pd.notna(qb_cov) else 0) - (qb_pr * (1-qb_cov_w) if pd.notna(qb_pr) else 0) if pd.notna(qb_off) else np.nan

    rb_run = def_map.get((def_team, "RunDefense"), np.nan)
    rb_cov = def_map.get((def_team, "CoverageLB"), np.nan)
    rb_off = off_map.get((off_team, "RB"), np.nan)
    rb_edge = rb_off - (rb_run * rb_run_w if pd.notna(rb_run) else 0) - (rb_cov * (1-rb_run_w) if pd.notna(rb_cov) else 0) if pd.notna(rb_off) else np.nan

    wr_cov = def_map.get((def_team, "CoverageDB"), np.nan)
    wr_off = off_map.get((off_team, "WR"), np.nan)
    wr_edge = wr_off - wr_cov if (pd.notna(wr_off) and pd.notna(wr_cov)) else np.nan

    te_covlb = def_map.get((def_team, "CoverageLB"), np.nan)
    te_covdb = def_map.get((def_team, "CoverageDB"), np.nan)
    te_off   = off_map.get((off_team, "TE"), np.nan)
    te_edge = te_off - (te_covlb * te_covlb_w if pd.notna(te_covlb) else 0) - (te_covdb * (1-te_covlb_w) if pd.notna(te_covdb) else 0) if pd.notna(te_off) else np.nan

    ol_pr  = def_map.get((def_team, "PassRush"), np.nan)
    ol_run = def_map.get((def_team, "RunDefense"), np.nan)
    ol_off = off_map.get((off_team, "OL"), np.nan)
    ol_edge = ol_off - (ol_pr * ol_pass_w if pd.notna(ol_pr) else 0) - (ol_run * (1-ol_pass_w) if pd.notna(ol_run) else 0) if pd.notna(ol_off) else np.nan

    return {"QB": qb_edge, "RB": rb_edge, "WR": wr_edge, "TE": te_edge, "OL": ol_edge}

def adjusted_team_edge(off_team, def_team, off_map, def_map, dep_strength, weights, qb_cov_w, rb_run_w, te_covlb_w, ol_pass_w):
    raw = unit_matchup(off_team, def_team, off_map, def_map, qb_cov_w, rb_run_w, te_covlb_w, ol_pass_w)
    ol_edge = raw["OL"]
    ttf = 1.0 if pd.isna(ol_edge) else np.clip(0.6 + 0.4 * ol_edge * dep_strength, 0.2, 1.0)
    adj = {
        "QB": raw["QB"] * ttf if pd.notna(raw["QB"]) else np.nan,
        "WR": raw["WR"] * ttf if pd.notna(raw["WR"]) else np.nan,
        "TE": raw["TE"] * ttf if pd.notna(raw["TE"]) else np.nan,
        "RB": raw["RB"],
        "OL": raw["OL"],
    }
    total, wsum = 0.0, 0.0
    for k,v in adj.items():
        if pd.notna(v) and weights[k] > 0:
            total += weights[k]*v
            wsum  += weights[k]
    team_edge = (total/wsum) if wsum>0 else np.nan
    return team_edge, raw, adj, ttf

sched = load_schedule()
pbp   = load_pbp(season)
off_u, def_u = build_units(pbp, season, week, espn_enable, sdio_enable)
of = unitize_offense(off_u); df = unitize_defense(def_u)
off_map, def_map = build_maps(of, df)

wk = sched[(sched["season"]==season) & (sched["week"]==week) & (sched["game_type"]=="REG")].copy()
if wk.empty:
    st.warning("No regular-season games found in the schedule for that week.")
else:
    wk["label"] = wk["away_team"] + " @ " + wk["home_team"] + "  (" + wk["gameday"].astype(str) + ")"
    st.subheader(f"Week {int(week)} — Picks")
    weights = {"QB":w_qb, "RB":w_rb, "WR":w_wr, "TE":w_te, "OL":w_ol}
    for _, g in wk.sort_values("gameday").iterrows():
        home, away = g["home_team"], g["away_team"]
        home_edge, home_raw, home_adj, home_ttf = adjusted_team_edge(home, away, off_map, def_map, dep_strength, weights, qb_cov_w, rb_run_w, te_covlb_w, ol_pass_w)
        away_edge, away_raw, away_adj, away_ttf = adjusted_team_edge(away, home, off_map, def_map, dep_strength, weights, qb_cov_w, rb_run_w, te_covlb_w, ol_pass_w)
        net = home_edge - away_edge if (pd.notna(home_edge) and pd.notna(away_edge)) else np.nan

        if pd.isna(net): verdict = "Insufficient data"
        elif net > close_margin: verdict = f"**{home} should win over {away}**"
        elif net < -close_margin: verdict = f"**{away} should win over {home}**"
        else: verdict = "**Too close to call**"

        st.markdown(f"### {g['label']} — {verdict}")

        with st.expander("See matchup breakdown"):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**{home} offense vs {away} defense**")
                df_home = pd.DataFrame({"unit":["QB","RB","WR","TE","OL"],
                                        "raw_edge":[home_raw["QB"],home_raw["RB"],home_raw["WR"],home_raw["TE"],home_raw["OL"]],
                                        "adjusted":[home_adj["QB"],home_adj["RB"],home_adj["WR"],home_adj["TE"],home_adj["OL"]]})
                st.dataframe(df_home, hide_index=True)
                st.caption(f"Pass-game scaling factor (TTF): {home_ttf:.2f}")
                st.metric("Overall adjusted edge (home offense)", f"{home_edge:+.3f}")
            with cols[1]:
                st.markdown(f"**{away} offense vs {home} defense**")
                df_away = pd.DataFrame({"unit":["QB","RB","WR","TE","OL"],
                                        "raw_edge":[away_raw["QB"],away_raw["RB"],away_raw["WR"],away_raw["TE"],away_raw["OL"]],
                                        "adjusted":[away_adj["QB"],away_adj["RB"],away_adj["WR"],away_adj["TE"],away_adj["OL"]]})
                st.dataframe(df_away, hide_index=True)
                st.caption(f"Pass-game scaling factor (TTF): {away_ttf:.2f}")
                st.metric("Overall adjusted edge (away offense)", f"{away_edge:+.3f}")

            st.markdown("---")
            st.metric("Net team edge (home - away)", f"{net:+.3f}")
            st.caption(f"Verdict threshold for 'close game' = ±{close_margin:.2f}")
