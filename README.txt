
NFL Matchup Picks — Live Data (nflverse) + Provider Adapters

This Streamlit app auto-fetches NFL schedule and team/unit metrics using the public
nflverse (nflfastR) datasets at runtime — no manual CSVs required. It still supports uploads.

It also ships with adapters you can enable for proprietary per-player win-rate data
(ESPN PRWR/PBWR, PFF, SportsDataIO). Wire your keys in st.secrets and flip the toggles
to enrich trench/coverage matchups automatically.

Quick start:
    pip install streamlit pandas numpy requests pyarrow fastparquet
    streamlit run streamlit_app.py
