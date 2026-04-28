import sys
import os
import streamlit as st
import importlib.util
import importlib.machinery
from typing import List, Dict, Any
import random
from collections import Counter
import pandas as pd


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


BASE = os.path.join(os.path.dirname(__file__), "ReliefLens-BDA", "bda features")

# Map desired relieflens.* names to actual files in the repo
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
        # ensure parent packages exist
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
    /* Align table content directly under column headers and collapse spacing */
    table.dataframe, .stDataFrame table, table {
        border-collapse: collapse !important;
        border-spacing: 0 !important;
    }
    table.dataframe thead th, table thead th, .stDataFrame table thead th {
        vertical-align: bottom !important;
        padding-top: 6px !important;
        padding-bottom: 2px !important;
    }
    table.dataframe tbody td, table tbody td, .stDataFrame table tbody td {
        vertical-align: top !important;
        padding-top: 2px !important;
        padding-bottom: 6px !important;
        line-height: 1.1 !important;
    }
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


if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "logs" not in st.session_state:
    st.session_state.logs = []


with st.sidebar.form(key="controls"):
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


def synthesize_predictions(
    n: int, regions: int, disasters: List[str], error_rate: float = 0.25
) -> (List[Any], List[Dict[str, Any]]):
    region_names = [f"Zone{chr(65+i)}" for i in range(regions)]
    preds = []
    logs = []
    ERR_CODES = ["ERR01", "ERR02", "ERR03", "ERR04"]
    for i in range(n):
        region = random.choice(region_names)
        disaster = (
            random.choice(disasters)
            if disasters
            else random.choice(["Flood", "Fire", "Earthquake", "Cyclone"])
        )
        # probabilities: 4 classes
        probs = [random.random() for _ in range(4)]
        s = sum(probs)
        probs = [p / s for p in probs]
        # predicted class biased by probabilities (argmax)
        pred_class = int(max(range(4), key=lambda j: probs[j]))
        P = rl_pipeline.Prediction(
            region=region,
            disaster_type=disaster,
            predicted_class=pred_class,
            probabilities=probs,
        )
        preds.append(P)
        # logs: randomly include some error codes
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
    st.success(f"Generated {len(preds)} predictions across {n_regions} regions.")


def safe_run_analytics(preds, logs):
    try:
        return rl_pipeline.run_analytics(preds, logs)
    except Exception as e:
        st.error(f"Analytics failed: {e}")
        return None


run_result = None
if run_analytics_btn:
    if not st.session_state.predictions:
        st.warning("Please generate data before running analytics.")
    else:
        with st.spinner("Running analytics pipeline..."):
            run_result = safe_run_analytics(
                st.session_state.predictions, st.session_state.logs
            )
        if run_result:
            st.success("Analytics complete.")


tabs = st.tabs(
    [
        "Overview",
        "MapReduce",
        "Top-K",
        "Threshold",
        "Spark Logs",
        "Cassandra",
        "MongoDB",
        "Export",
    ]
)

# Overview
with tabs[0]:
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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
    # Bar chart predictions per region
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
    else:
        st.info("No data generated yet. Use the sidebar to generate data.")

# MapReduce
with tabs[1]:
    st.subheader("🔁 MapReduce")
    if run_result and run_result.get("avg_severity"):
        table_rows = [
            {"region": k[0], "disaster_type": k[1], "avg_severity": v}
            for k, v in run_result["avg_severity"].items()
        ]
        st.dataframe(pd.DataFrame(table_rows))
        complexities = rl_mapreduce.compute_complexity()
        st.markdown("**Complexity**")
        st.write(complexities)
        with st.expander("How MapReduce Works"):
            st.markdown(
                "MapReduce maps each prediction to a (region,disaster) key, shuffles and groups values, then reduces by averaging severity. This simulates distributed aggregation across partitions."
            )
    else:
        st.info("Run analytics to see MapReduce results.")

# Top-K
with tabs[2]:
    st.subheader("🔥 Top-K Hotspots")
    if st.session_state.predictions:
        k = top_k
        topk_list = rl_topk.top_k_regions(st.session_state.predictions, k=k)
        if topk_list:
            dfk = pd.DataFrame(topk_list, columns=["region", "count"]).set_index(
                "region"
            )
            st.bar_chart(dfk)
            st.dataframe(dfk)
        else:
            st.info("No hotspots detected")
    else:
        st.info("Generate data to compute Top-K hotspots.")

# Threshold
with tabs[3]:
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

# Spark Logs
with tabs[4]:
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
    else:
        st.info("No logs available. Generate data with logs enabled.")

# Cassandra
with tabs[5]:
    st.subheader("🗄️ Cassandra (Simulation)")
    st.code(rl_cassandra.CQL_SCHEMA, language="sql")
    if run_result and run_result.get("cassandra_snapshot"):
        df_cas = pd.DataFrame(run_result["cassandra_snapshot"]).fillna("")
        st.dataframe(df_cas)
        st.markdown(
            "**TTL concept:** rows can be inserted with a TTL and will expire; this simulation stores an `expires_at` timestamp when TTL is used."
        )
    else:
        st.info(
            "Run analytics to write aggregated records to the Cassandra simulation."
        )

# MongoDB
with tabs[6]:
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
        total = mongo.count_total("predictions")
        st.metric("Total records", total)
        counts = mongo.count_by_field("predictions", "region")
        st.dataframe(
            pd.DataFrame(list(counts.items()), columns=["region", "count"]).set_index(
                "region"
            )
        )
    else:
        st.info("Generate data to populate the MongoDB simulation.")

# Export
with tabs[7]:
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


with st.expander("Preview Data & Logs"):
    st.markdown("**First 10 predictions**")
    if st.session_state.predictions:
        df_preview = pd.DataFrame(
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
        st.dataframe(df_preview)
    else:
        st.write("No predictions yet.")

    st.markdown("**Logs preview (first 10 entries)**")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs[:10]))
    else:
        st.write("No logs yet.")
