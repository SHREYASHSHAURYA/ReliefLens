import sys
import os
import streamlit as st
import importlib.util
import importlib.machinery
from typing import List, Dict, Any
import random
from collections import Counter
import pandas as pd
import io
import csv
import datetime
import json as _json


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


BASE = os.path.join(os.path.dirname(__file__), "ReliefLens-BDA", "bda features")

MODULE_MAP = {
    "relieflens.analytics.threshold": os.path.join(BASE, "analytics", "threshold.py"),
    "relieflens.analytics.topk": os.path.join(BASE, "analytics", "topk.py"),
    "relieflens.data_processing.mapreduce": os.path.join(
        BASE, "data_processing", "mapreduce.py"
    ),
    "relieflens.data_processing.spark_sim": os.path.join(
        BASE, "data_processing", "spark_sim.py"
    ),
    "relieflens.exports.csv_io": os.path.join(BASE, "exports", "csv_io.py"),
    "relieflens.storage.cassandra_sim": os.path.join(
        BASE, "storage", "cassandra_sim.py"
    ),
    "relieflens.storage.mongo_sim": os.path.join(BASE, "storage", "mongo_sim.py"),
    "relieflens.pipeline": os.path.join(BASE, "pipeline.py"),
}


def ensure_modules():
    import types

    for name, path in MODULE_MAP.items():
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                pkg = types.ModuleType(parent)
                pkg.__path__ = []
                sys.modules[parent] = pkg
        if name in sys.modules:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required module file not found: {path}")
        load_module_from_path(name, path)


ensure_modules()

from relieflens import pipeline as rl_pipeline  # type: ignore
from relieflens.data_processing import mapreduce as rl_mapreduce  # type: ignore
from relieflens.data_processing import spark_sim as rl_spark  # type: ignore
from relieflens.analytics import topk as rl_topk  # type: ignore
from relieflens.analytics import threshold as rl_threshold  # type: ignore
from relieflens.storage import cassandra_sim as rl_cassandra  # type: ignore
from relieflens.storage import mongo_sim as rl_mongo  # type: ignore
from relieflens.exports import csv_io as rl_csv  # type: ignore


st.set_page_config(
    page_title="ReliefLens Analytics Dashboard", layout="wide", page_icon="📊"
)

st.markdown(
    """
    <style>
    :root { --bg: #fff6f2; --panel: #ffffff; --accent: #ff3b2e; --muted: #6b7280; }
    .stApp, .block-container { background: var(--bg); color: #0f1724; }
    .title {font-size:34px; font-weight:700; color: var(--accent);}
    .subtitle {color: var(--muted); margin-top: -10px}
    .card {background: var(--panel); padding:16px; border-radius:12px; border: 1px solid rgba(15,23,36,0.06); box-shadow: 0 6px 18px rgba(15,23,36,0.04)}
    .stSidebar .sidebar-content { background: var(--panel); }
    .stMetricValue { color: #0f1724 !important; }
    .dataframe, table { background: var(--panel) }
    table.dataframe, .stDataFrame table, table { border-collapse: collapse !important; border-spacing: 0 !important; }
    table.dataframe thead th, table thead th, .stDataFrame table thead th { vertical-align: bottom !important; padding-top: 6px !important; padding-bottom: 2px !important; }
    table.dataframe tbody td, table tbody td, .stDataFrame table tbody td { vertical-align: top !important; padding-top: 2px !important; padding-bottom: 6px !important; line-height: 1.1 !important; }
    .stDataFrame table tbody tr td:first-child { padding-left: 10px !important; }
    .stDataFrame table thead th { padding-left: 10px !important; }
    .stButton>button { background-color: var(--accent); color: #fff; border: none }
    </style>
""",
    unsafe_allow_html=True,
)

with st.container():
    st.markdown(
        '<div class="title">ReliefLens Analytics Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Interactive simulation of MapReduce, Spark, and NoSQL analytics on disaster data</div>',
        unsafe_allow_html=True,
    )

# ── Session state defaults ────────────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "run_result" not in st.session_state:
    st.session_state.run_result = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Navigation radio — THIS is what replaces st.tabs.
    # session_state key "active_tab" persists across every rerun automatically.
    st.header("Navigate")
    TAB_NAMES = [
        "Overview",
        "MapReduce",
        "Top-K",
        "Threshold",
        "Spark Logs",
        "Cassandra",
        "MongoDB",
        "Export",
    ]
    selected_tab = st.radio(
        "", TAB_NAMES, key="active_tab", label_visibility="collapsed"
    )

    st.divider()

    with st.form(key="controls"):
        st.header("Control Panel")
        n_preds = st.slider("Number of predictions", 10, 100, 50)
        n_regions = st.slider("Number of regions", 2, 6, 3)
        disaster_options = ["Flood", "Fire", "Earthquake", "Cyclone"]
        disasters = st.multiselect(
            "Disaster types", disaster_options, default=disaster_options[:2]
        )
        threshold_val = st.slider("Threshold for severe detection", 0.3, 0.9, 0.7, 0.01)
        top_k = st.slider("Top-K value", 3, 15, 5)
        error_rate = st.slider("Error log rate", 0.0, 1.0, 0.25, 0.05)
        generate = st.form_submit_button("Generate Data")
        run_analytics_btn = st.form_submit_button("Run Analytics")


def synthesize_predictions(n, regions, disasters, error_rate=0.25):
    region_names = [f"Zone{chr(65+i)}" for i in range(regions)]
    preds, logs = [], []
    ERR_CODES = ["ERR01", "ERR02", "ERR03", "ERR04"]
    for i in range(n):
        region = random.choice(region_names)
        disaster = (
            random.choice(disasters)
            if disasters
            else random.choice(["Flood", "Fire", "Earthquake", "Cyclone"])
        )
        probs = [random.random() for _ in range(4)]
        s = sum(probs)
        probs = [p / s for p in probs]
        pred_class = int(max(range(4), key=lambda j: probs[j]))
        P = rl_pipeline.Prediction(
            region=region,
            disaster_type=disaster,
            predicted_class=pred_class,
            probabilities=probs,
        )
        preds.append(P)
        if random.random() < error_rate:
            logs.append(
                {
                    "errors": [
                        random.choice(ERR_CODES) for _ in range(random.randint(1, 4))
                    ]
                }
            )
        else:
            logs.append({"errors": []})
    return preds, logs


if generate:
    preds, logs = synthesize_predictions(
        n_preds, n_regions, disasters, error_rate=error_rate
    )
    st.session_state.predictions = preds
    st.session_state.logs = logs
    st.session_state.run_result = None  # reset stale analytics on new data
    st.success(f"Generated {len(preds)} predictions across {n_regions} regions.")

if run_analytics_btn:
    if not st.session_state.predictions:
        st.warning("Please generate data before running analytics.")
    else:
        with st.spinner("Running analytics pipeline..."):
            try:
                st.session_state.run_result = rl_pipeline.run_analytics(
                    st.session_state.predictions, st.session_state.logs
                )
            except Exception as e:
                st.error(f"Analytics failed: {e}")
                st.session_state.run_result = None
        if st.session_state.run_result:
            st.success("Analytics complete.")

# Convenience alias so tab code reads cleanly
run_result = st.session_state.run_result

# ── Tab content — plain if/elif, no st.tabs() ────────────────────────────────

if selected_tab == "Overview":
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_preds = len(st.session_state.predictions)
    severe_ct = (
        rl_threshold.count_severe(st.session_state.predictions, threshold=threshold_val)
        if st.session_state.predictions
        else 0
    )
    unique_regions = (
        len({p.region for p in st.session_state.predictions})
        if st.session_state.predictions
        else 0
    )
    logs_with_errors = (
        sum(1 for e in st.session_state.logs if e.get("errors"))
        if st.session_state.logs
        else 0
    )
    col1.metric("Total Predictions", total_preds)
    col2.metric("Severe Cases", severe_ct)
    col3.metric("Unique Regions", unique_regions)
    col4.metric("Logs w/ errors", logs_with_errors)
    st.markdown("##")
    if st.session_state.predictions:
        df = pd.DataFrame(
            [
                {
                    "region": p.region,
                    "disaster": p.disaster_type,
                    "severity": p.severity_score,
                }
                for p in st.session_state.predictions
            ]
        )
        region_counts = df.groupby("region").size().reset_index(name="count")
        st.bar_chart(data=region_counts.set_index("region"))
        st.markdown("---")
        dcol1, dcol2 = st.columns([2, 1])
        with dcol1:
            st.markdown("**Disaster breakdown**")
            disaster_counts = df.groupby("disaster").size().reset_index(name="count")
            st.dataframe(disaster_counts.set_index("disaster"))
            st.bar_chart(data=disaster_counts.set_index("disaster"))
        with dcol2:
            st.markdown("**Region details**")
            regions = sorted(df["region"].unique())
            sel = st.selectbox("Select region", ["(all)"] + regions)
            if sel and sel != "(all)":
                rr = df[df["region"] == sel]
                st.metric("Predictions", len(rr))
                st.metric("Avg severity", float(rr["severity"].mean()))
                st.dataframe(rr)
    else:
        st.info("No data generated yet. Use the sidebar to generate data.")

elif selected_tab == "MapReduce":
    st.subheader("🔁 MapReduce")
    if run_result and run_result.get("avg_severity"):
        table_rows = [
            {"region": k[0], "disaster_type": k[1], "avg_severity": v}
            for k, v in run_result["avg_severity"].items()
        ]
        df_map = pd.DataFrame(table_rows)

        # ── Summary metrics ───────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total partitions", len(table_rows))
        mc2.metric("Unique regions", df_map["region"].nunique())
        mc3.metric("Unique disaster types", df_map["disaster_type"].nunique())
        mc4.metric("Overall avg severity", f"{df_map['avg_severity'].mean():.3f}")
        st.markdown("---")

        # ── Interactive filters ───────────────────────────────────────────────
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            region_filter = st.multiselect(
                "Filter by region",
                options=sorted(df_map["region"].unique()),
                default=sorted(df_map["region"].unique()),
                key="mr_region_filter",
            )
        with fcol2:
            disaster_filter = st.multiselect(
                "Filter by disaster type",
                options=sorted(df_map["disaster_type"].unique()),
                default=sorted(df_map["disaster_type"].unique()),
                key="mr_disaster_filter",
            )
        severity_range = st.slider(
            "Avg severity range",
            min_value=float(df_map["avg_severity"].min()),
            max_value=float(df_map["avg_severity"].max()),
            value=(
                float(df_map["avg_severity"].min()),
                float(df_map["avg_severity"].max()),
            ),
            key="mr_severity_slider",
        )

        df_filtered = df_map[
            df_map["region"].isin(region_filter)
            & df_map["disaster_type"].isin(disaster_filter)
            & df_map["avg_severity"].between(*severity_range)
        ]
        st.markdown(f"**Showing {len(df_filtered)} of {len(df_map)} partitions**")
        st.dataframe(df_filtered.reset_index(drop=True))

        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Avg severity by disaster type**")
            st.bar_chart(df_filtered.groupby("disaster_type")["avg_severity"].mean())
        with ch2:
            st.markdown("**Avg severity by region**")
            st.bar_chart(df_filtered.groupby("region")["avg_severity"].mean())

        # ── Complexity + how-it-works ─────────────────────────────────────────
        st.markdown("---")
        complexities = rl_mapreduce.compute_complexity()
        st.markdown("**Complexity**")
        st.write(complexities)
        with st.expander("How MapReduce Works"):
            st.markdown(
                "MapReduce maps each prediction to a (region,disaster) key, shuffles and groups values, then reduces by averaging severity. This simulates distributed aggregation across partitions."
            )
    else:
        st.info("Run analytics to see MapReduce results.")

elif selected_tab == "Top-K":
    st.subheader("🔥 Top-K Hotspots")
    if st.session_state.predictions:
        topk_list = rl_topk.top_k_regions(st.session_state.predictions, k=top_k)
        if topk_list:
            df_all = pd.DataFrame(
                [
                    {
                        "region": p.region,
                        "disaster_type": p.disaster_type,
                        "severity": p.severity_score,
                    }
                    for p in st.session_state.predictions
                ]
            )
            dfk = pd.DataFrame(topk_list, columns=["region", "count"])

            # ── Summary metrics ───────────────────────────────────────────────
            total_in_hotspots = dfk["count"].sum()
            total_preds = len(st.session_state.predictions)
            top1_region = dfk.iloc[0]["region"]
            top1_count = int(dfk.iloc[0]["count"])
            avg_sev_top1 = df_all[df_all["region"] == top1_region]["severity"].mean()

            km1, km2, km3, km4 = st.columns(4)
            km1.metric("Top hotspot", top1_region)
            km2.metric(f"{top1_region} predictions", top1_count)
            km3.metric(f"{top1_region} avg severity", f"{avg_sev_top1:.3f}")
            km4.metric(
                f"Top-{top_k} share of total",
                f"{total_in_hotspots/total_preds*100:.1f}%",
            )
            st.markdown("---")

            # ── Interactive N slicer ──────────────────────────────────────────
            max_k = len(topk_list)
            display_k = st.slider(
                "Show top N regions",
                min_value=1,
                max_value=max_k,
                value=max_k,
                key="topk_display_k",
            )
            dfk_display = dfk.head(display_k).set_index("region")

            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Prediction count by region**")
                st.bar_chart(dfk_display)
            with ch2:
                st.markdown("**Avg severity by region**")
                sev_by_region = (
                    df_all[df_all["region"].isin(dfk_display.index)]
                    .groupby("region")["severity"]
                    .mean()
                    .reindex(dfk_display.index)
                )
                st.bar_chart(sev_by_region)

            # ── Cumulative share ──────────────────────────────────────────────
            st.markdown("**Cumulative prediction share across top-N regions**")
            dfk_cum = dfk.copy()
            dfk_cum["cumulative_%"] = (
                dfk_cum["count"].cumsum() / total_preds * 100
            ).round(1)
            dfk_cum.index = range(1, len(dfk_cum) + 1)
            st.line_chart(dfk_cum["cumulative_%"])

            # ── Per-hotspot disaster breakdown ────────────────────────────────
            st.markdown("**Disaster type breakdown for a selected hotspot**")
            breakdown_filter = st.selectbox(
                "Select region to inspect",
                options=[r for r, _ in topk_list],
                key="topk_region_inspect",
            )
            region_df = df_all[df_all["region"] == breakdown_filter]
            dis_counts = (
                region_df.groupby("disaster_type").size().reset_index(name="count")
            )
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.dataframe(dis_counts.set_index("disaster_type"))
            with bcol2:
                st.bar_chart(dis_counts.set_index("disaster_type"))

            st.markdown("---")
            st.dataframe(dfk_display)
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["region", "count"])
            for r, c in topk_list:
                writer.writerow([r, c])
            st.download_button(
                "Download Top-K CSV",
                data=csv_buf.getvalue(),
                file_name="topk_hotspots.csv",
                mime="text/csv",
            )
        else:
            st.info("No hotspots detected")
    else:
        st.info("Generate data to compute Top-K hotspots.")

elif selected_tab == "Threshold":
    st.subheader("⚠️ Threshold Analysis")
    if st.session_state.predictions:
        severe_count = rl_threshold.count_severe(
            st.session_state.predictions, threshold=threshold_val
        )
        pct = (
            (severe_count / len(st.session_state.predictions) * 100)
            if st.session_state.predictions
            else 0
        )
        st.metric("Severe count", severe_count, delta=f"{pct:.1f}% of predictions")
        st.markdown("##")
        pie_df = pd.DataFrame(
            {
                "label": ["Severe", "Non-Severe"],
                "value": [
                    severe_count,
                    len(st.session_state.predictions) - severe_count,
                ],
            }
        )
        st.altair_chart(
            (
                __import__("altair")
                .Chart(pie_df)
                .mark_arc()
                .encode(theta="value:Q", color="label:N")
            ),
            use_container_width=True,
        )
    else:
        st.info("Generate data to run threshold analysis.")

elif selected_tab == "Spark Logs":
    st.subheader("🟣 Spark Logs")
    if st.session_state.logs:
        all_errors = []
        for entry in st.session_state.logs:
            all_errors.extend(entry.get("errors", []))
        counts = Counter(all_errors)
        df_err = pd.DataFrame(
            list(counts.items()), columns=["error", "count"]
        ).sort_values("count", ascending=False)
        if not df_err.empty:
            df_err["flag"] = df_err["count"].apply(lambda x: "⚠️" if x > 10 else "")
            st.dataframe(df_err)
        else:
            st.info("No errors recorded in logs.")
        with st.expander("About Spark-style log processing"):
            st.write(
                "This simulates flatMap -> map -> reduceByKey -> filter. Only errors with counts > 10 are considered critical in our filter step."
            )
        with st.expander("Log exploration"):
            query = st.text_input("Filter error code (e.g. ERR01)")
            timeline = [
                {"index": i, "n_errors": len(entry.get("errors", []))}
                for i, entry in enumerate(st.session_state.logs)
            ]
            tdf = pd.DataFrame(timeline)
            if not tdf.empty:
                st.line_chart(tdf.set_index("index"))
            if query:
                filtered = [
                    e
                    for e in st.session_state.logs
                    if query in (" ".join(e.get("errors", [])))
                ]
                st.markdown(f"Found {len(filtered)} entries matching '{query}'")
                if filtered:
                    st.dataframe(pd.DataFrame(filtered))
    else:
        st.info("No logs available. Generate data with logs enabled.")

elif selected_tab == "Cassandra":
    st.subheader("🗄️ Cassandra (Simulation)")
    st.code(rl_cassandra.CQL_SCHEMA, language="sql")
    if run_result and run_result.get("cassandra_snapshot"):
        df_cas_raw = pd.DataFrame(run_result["cassandra_snapshot"]).fillna("")

        # ── Summary metrics ───────────────────────────────────────────────────
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Total rows", len(df_cas_raw))
        cm2.metric("Columns", len(df_cas_raw.columns))
        numeric_cols = df_cas_raw.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            cm3.metric(
                f"Avg {numeric_cols[0]}",
                f"{pd.to_numeric(df_cas_raw[numeric_cols[0]], errors='coerce').mean():.3f}",
            )
        st.markdown("---")

        # ── Interactive column selector ───────────────────────────────────────
        all_cols = df_cas_raw.columns.tolist()
        selected_cols = st.multiselect(
            "Choose columns to display",
            options=all_cols,
            default=all_cols,
            key="cas_col_select",
        )
        df_cas = df_cas_raw[selected_cols] if selected_cols else df_cas_raw

        # ── Sort control ──────────────────────────────────────────────────────
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_by = st.selectbox(
                "Sort by column", options=["(none)"] + selected_cols, key="cas_sort_col"
            )
        with sort_col2:
            sort_asc = st.radio(
                "Order",
                ["Ascending", "Descending"],
                horizontal=True,
                key="cas_sort_order",
            )
        if sort_by != "(none)":
            df_cas = df_cas.sort_values(sort_by, ascending=(sort_asc == "Ascending"))

        # ── Row search ────────────────────────────────────────────────────────
        row_search = st.text_input(
            "Search rows (matches any column value, case-insensitive)",
            key="cas_row_search",
        )
        if row_search:
            mask = df_cas.apply(
                lambda col: col.astype(str).str.contains(
                    row_search, case=False, na=False
                )
            ).any(axis=1)
            df_cas = df_cas[mask]
            st.markdown(f"**{len(df_cas)} rows match '{row_search}'**")

        st.dataframe(df_cas.reset_index(drop=True))

        # ── Numeric column chart ──────────────────────────────────────────────
        if numeric_cols:
            chart_col = st.selectbox(
                "Plot distribution of", options=numeric_cols, key="cas_chart_col"
            )
            chart_series = pd.to_numeric(
                df_cas_raw[chart_col], errors="coerce"
            ).dropna()
            if not chart_series.empty:
                st.markdown(f"**Distribution of `{chart_col}`**")
                hist_df = pd.cut(chart_series, bins=10).value_counts().sort_index()
                hist_df = hist_df.reset_index()
                hist_df.columns = ["bin", "count"]
                hist_df["bin"] = hist_df["bin"].astype(str)
                st.bar_chart(hist_df.set_index("bin"))

        st.markdown("---")
        st.markdown(
            "**TTL concept:** rows can be inserted with a TTL and will expire; this simulation stores an `expires_at` timestamp when TTL is used."
        )
        with st.expander("Cassandra TTL simulator"):
            try:
                now = datetime.datetime.utcnow().timestamp()
                df_ttl = df_cas_raw.copy()
                df_ttl["expires_at"] = pd.to_numeric(
                    df_ttl.get("expires_at", pd.Series([None] * len(df_ttl))),
                    errors="coerce",
                )
                alive = df_ttl[
                    df_ttl["expires_at"].isna() | (df_ttl["expires_at"] > now)
                ]
                expired = df_ttl[
                    ~(df_ttl["expires_at"].isna() | (df_ttl["expires_at"] > now))
                ]
                st.markdown(
                    f"**Alive rows:** {len(alive)} — **Expired rows:** {len(expired)}"
                )
                if st.button("Show expired rows"):
                    st.dataframe(expired)
            except Exception:
                st.write("TTL simulator not available for this snapshot.")
    else:
        st.info(
            "Run analytics to write aggregated records to the Cassandra simulation."
        )

elif selected_tab == "MongoDB":
    st.subheader("🟠 MongoDB (Simulation)")
    mongo = rl_mongo.MongoSim()
    if st.session_state.predictions:
        for p in st.session_state.predictions:
            mongo.insert(
                "predictions",
                {
                    "region": p.region,
                    "disaster_type": p.disaster_type,
                    "severity": p.severity_score,
                },
            )

        docs_all = mongo.find_all("predictions")
        df_mongo = pd.DataFrame(docs_all) if docs_all else pd.DataFrame()

        # ── Summary metrics ───────────────────────────────────────────────────
        total = mongo.count_total("predictions")
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Total records", total)
        mm2.metric(
            "Unique regions", df_mongo["region"].nunique() if not df_mongo.empty else 0
        )
        mm3.metric(
            "Unique disaster types",
            df_mongo["disaster_type"].nunique() if not df_mongo.empty else 0,
        )
        mm4.metric(
            "Avg severity",
            (
                f"{pd.to_numeric(df_mongo['severity'], errors='coerce').mean():.3f}"
                if not df_mongo.empty
                else "—"
            ),
        )
        st.markdown("---")

        # ── Charts: counts and severity by region/disaster ────────────────────
        if not df_mongo.empty:
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Records per region**")
                counts_region = mongo.count_by_field("predictions", "region")
                st.bar_chart(
                    pd.DataFrame(
                        list(counts_region.items()), columns=["region", "count"]
                    ).set_index("region")
                )
            with ch2:
                st.markdown("**Records per disaster type**")
                counts_dis = mongo.count_by_field("predictions", "disaster_type")
                st.bar_chart(
                    pd.DataFrame(
                        list(counts_dis.items()), columns=["disaster_type", "count"]
                    ).set_index("disaster_type")
                )

            # Severity distribution filter
            st.markdown("---")
            st.markdown("**Severity distribution explorer**")
            sev_series = pd.to_numeric(df_mongo["severity"], errors="coerce").dropna()
            sev_min, sev_max = float(sev_series.min()), float(sev_series.max())
            sev_range = st.slider(
                "Filter by severity range",
                min_value=sev_min,
                max_value=sev_max,
                value=(sev_min, sev_max),
                key="mongo_sev_slider",
            )
            df_sev_filtered = df_mongo[
                pd.to_numeric(df_mongo["severity"], errors="coerce").between(*sev_range)
            ]
            st.markdown(
                f"**{len(df_sev_filtered)} records** in severity range `{sev_range[0]:.3f}` – `{sev_range[1]:.3f}`"
            )

            agg_col1, agg_col2 = st.columns(2)
            with agg_col1:
                st.markdown("**Avg severity by region (filtered)**")
                agg_region = (
                    df_sev_filtered.groupby("region")["severity"]
                    .mean()
                    .apply(lambda x: round(float(x), 3))
                )
                st.dataframe(agg_region.rename("avg_severity"))
            with agg_col2:
                st.markdown("**Avg severity by disaster (filtered)**")
                agg_dis = (
                    df_sev_filtered.groupby("disaster_type")["severity"]
                    .mean()
                    .apply(lambda x: round(float(x), 3))
                )
                st.dataframe(agg_dis.rename("avg_severity"))

        st.markdown("---")

        with st.expander("Query & export predictions"):
            st.markdown("Examples: `{'region':'ZoneA'}`, `{'disaster_type':'Flood'}`")
            q_text = st.text_input(
                "Query (JSON, leave blank for all)", value="", key="mongo_query_text"
            )
            sample_examples = st.selectbox(
                "Quick examples",
                (
                    "--none--",
                    "{'region':'ZoneA'}",
                    "{'disaster_type':'Flood'}",
                    "{'region':'ZoneA','disaster_type':'Flood'}",
                ),
                key="mongo_query_examples",
            )
            if sample_examples != "--none--" and q_text == "":
                q_text = sample_examples

            if st.button("Run query", key="mongo_run_query"):
                if q_text:
                    try:
                        qobj = _json.loads(q_text.replace("'", '"'))

                        def matches(doc, q):
                            return all(doc.get(k) == v for k, v in q.items())

                        st.session_state["mongo_query_results"] = [
                            d for d in docs_all if matches(d, qobj)
                        ]
                    except Exception as e:
                        st.error(f"Invalid query JSON: {e}")
                        st.session_state["mongo_query_results"] = []
                else:
                    st.session_state["mongo_query_results"] = docs_all
                st.session_state["mongo_query_last"] = (
                    datetime.datetime.utcnow().isoformat()
                )

            display_docs = st.session_state.get("mongo_query_results", docs_all)

            if display_docs:
                st.dataframe(pd.DataFrame(display_docs))
                if st.session_state.get("mongo_query_last"):
                    st.markdown(
                        f"**Last query run:** {st.session_state['mongo_query_last']}"
                    )
            else:
                st.write("No documents match the query.")

            st.download_button(
                "Download JSON of current query results",
                data=_json.dumps(display_docs, default=str, indent=2),
                file_name="predictions.json",
                mime="application/json",
                key="mongo_download_json",
            )
    else:
        st.info("Generate data to populate the MongoDB simulation.")

elif selected_tab == "Export":
    st.subheader("📤 Export")
    if (
        run_result
        and run_result.get("export_file")
        and os.path.exists(run_result["export_file"])
    ):
        fn = run_result["export_file"]
        st.write("CSV file:", fn)
        with open(fn, "rb") as fh:
            st.download_button(
                "Download CSV", fh, file_name=os.path.basename(fn), mime="text/csv"
            )
    else:
        st.info("No export available yet. Run analytics to generate CSV.")

# ── Preview expander (always visible) ────────────────────────────────────────
with st.expander("Preview Data & Logs"):
    st.markdown("**First 10 predictions**")
    if st.session_state.predictions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "region": p.region,
                        "disaster": p.disaster_type,
                        "predicted_class": p.predicted_class,
                        "probabilities": p.probabilities,
                    }
                    for p in st.session_state.predictions[:10]
                ]
            )
        )
    else:
        st.write("No predictions yet.")
    st.markdown("**Logs preview (first 10 entries)**")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs[:10]))
    else:
        st.write("No logs yet.")
