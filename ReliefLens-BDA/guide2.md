# ReliefLens Big Data Analytics Guide

This guide explains the full ReliefLens project with the new Big Data-style simulation layer added to the repository. It is designed for a course presentation, covering end-to-end architecture, the additional analytics layer, and how to verify and run everything.

---

## 1. Project Overview

ReliefLens is a disaster image classification system that detects structural damage severity from aerial, satellite, and drone images. The core ML pipeline is intact, and a new local simulation layer was added for Big Data-style analytics without requiring Hadoop or Spark.

### What was added

A new package was added under `relieflens/`:

- `relieflens/data_processing/`
  - `mapreduce.py` — MapReduce simulation for averaging severity by region and disaster type
  - `spark_sim.py` — Spark-style pipeline simulation for error log analysis
- `relieflens/analytics/`
  - `topk.py` — Top-K hotspot detection for severe damage regions
  - `threshold.py` — Severe-case threshold detection
- `relieflens/storage/`
  - `cassandra_sim.py` — Cassandra-like storage simulation with TTL and CQL schema
  - `mongo_sim.py` — MongoDB-like collection and counter simulation
- `relieflens/exports/`
  - `csv_io.py` — CSV export/import utilities
- `relieflens/pipeline.py` — Integration pipeline that ties all components together

These files are modular, local, and runnable with plain Python.

---

## 2. Why this addition exists

The new module layer simulates Big Data analytics concepts for the ReliefLens project:

- **MapReduce** for regional severity aggregation
- **Spark-style transformations** for log analysis
- **Cassandra simulation** for storing aggregated monitoring data
- **MongoDB simulation** for document and counter-style operations
- **CSV export/import** for simple data exchange

This is ideal for a Big Data analytics course because it shows how distributed concepts can be expressed in a local Python project without external frameworks.

---

## 3. New module explanations

### 3.1 `relieflens/data_processing/mapreduce.py`

This module simulates the three phases of MapReduce:

- **Map phase**
  - `map_phase(predictions)` emits exactly one key-value pair per prediction.
  - Key = `(region, disaster_type)`.
  - Value = `severity_score` (equal to the predicted class).
  - Complexity: `O(N)`.

- **Shuffle/Sort phase**
  - `shuffle_sort(mapped_data)` groups values by key.
  - Simulates data redistribution across partitions.
  - Complexity: `O(N log N)`.
  - Important note: In distributed systems, shuffle cost is dominated by network transfer rather than computation.

- **Reduce phase**
  - `reduce_phase(grouped_data)` computes average severity per key.
  - Complexity per key: `O(k)`.
  - Total reduce complexity: `O(N)`.

- **compute_complexity()**
  - Returns a dictionary of complexities:
    - Map: `O(N)`
    - Shuffle: `O(N log N)`
    - Reduce: `O(N)`
    - Overall: `O(N log N)`

### 3.2 `relieflens/analytics/topk.py`

This module implements hotspot detection with a Top-K algorithm:

- `top_k_regions(predictions, k=10)`
- Counts predictions with severity class >= 2 for each region
- Sorts regions by severe prediction count descending
- Returns the top `k` regions

### 3.3 `relieflens/data_processing/spark_sim.py`

This module simulates a Spark-style pipeline:

- `error_analysis(logs)`:
  - `flatMap`: extract all errors from logs
  - `map`: turn each error into `(error, 1)`
  - `reduceByKey`: count each error code
  - `filter`: return only errors with count > 10

This is a direct analogue of Spark transformations for error pattern detection.

### 3.4 `relieflens/analytics/threshold.py`

This module implements threshold-based severe detection:

- `detect_severe(prediction, threshold=0.7)`
  - Computes `severe_score = prob[2] + prob[3]`
  - Returns `True` if the score is above the threshold
- `count_severe(predictions, threshold=0.7)` counts how many predictions exceed the threshold

This matches existing ReliefLens logic where severe risk is identified by summing the major and destroyed probabilities.

### 3.5 `relieflens/storage/cassandra_sim.py`

This module simulates Cassandra-style storage and includes actual CQL schema text.

- `CQL_SCHEMA` string:
  - Creates `ReliefLens` keyspace
  - Creates `DamageMonitoring` table
- `CassandraSim` class:
  - `insert_batch(records)`
  - `update(region, disaster_type, new_value)`
  - `add_column(column_name)`
  - `insert_with_ttl(record, ttl_seconds)`
  - `query_all()`

It also simulates TTL using `expires_at` timestamps.

### 3.6 `relieflens/storage/mongo_sim.py`

This module simulates MongoDB collections and counters:

- `insert(collection, document)`
- `increment_counter(collection, key, amount=1)`
- `count_total(collection)`
- `count_by_field(collection, field)`

This is useful for demonstrating document storage and aggregation behavior in a course.

### 3.7 `relieflens/exports/csv_io.py`

This module provides file export/import:

- `export_csv(data, filename)`
- `import_csv(filename)`

It is the simplest way to persist results from the analytics pipeline.

### 3.8 `relieflens/pipeline.py`

This is the integration point for the new analytics layer.

- `Prediction` dataclass
- `run_analytics(predictions, logs)` performs:
  1. MapReduce average severity
  2. Top-K hotspot detection
  3. Threshold-based severe count
  4. Spark-style error analysis
  5. Store aggregated results in `CassandraSim`
  6. Export aggregated results to `relieflens_avg_severity.csv`

It returns a dictionary with:

- `avg_severity`
- `top_regions`
- `severe_cases`
- `error_patterns`
- `cassandra_snapshot`
- `export_file`

---

## 4. How to check everything

### 4.1 Verify package structure

Ensure the new package exists:

```powershell
Get-ChildItem relieflens -Recurse
```

You should see:

- `relieflens\data_processing\mapreduce.py`
- `relieflens\data_processing\spark_sim.py`
- `relieflens\analytics\topk.py`
- `relieflens\analytics\threshold.py`
- `relieflens\storage\cassandra_sim.py`
- `relieflens\storage\mongo_sim.py`
- `relieflens\exports\csv_io.py`
- `relieflens\pipeline.py`

### 4.2 Run Python syntax checks

```powershell
python -m py_compile relieflens\data_processing\mapreduce.py relieflens\data_processing\spark_sim.py relieflens\analytics\topk.py relieflens\analytics\threshold.py relieflens\storage\cassandra_sim.py relieflens\storage\mongo_sim.py relieflens\exports\csv_io.py relieflens\pipeline.py
```

If that command returns with no output and no error, the files are syntactically valid.

### 4.3 Run the full simulation pipeline

```powershell
python -c "from relieflens.pipeline import Prediction, run_analytics; predictions=[Prediction(region='ZoneA', disaster_type='Flood', predicted_class=3, probabilities=[0.0,0.0,0.1,0.9]), Prediction(region='ZoneA', disaster_type='Flood', predicted_class=2, probabilities=[0.0,0.0,0.6,0.4]), Prediction(region='ZoneB', disaster_type='Fire', predicted_class=1, probabilities=[0.2,0.7,0.05,0.05])]; logs=[{'errors':['ERR01','ERR02']},{'errors':['ERR01','ERR01','ERR03']},{'errors':['ERR02']}]; print(run_analytics(predictions, logs))"
```

This verifies the complete new layer end to end.

### 4.4 Verify CSV export

After running `run_analytics`, check for `relieflens_avg_severity.csv`:

```powershell
Get-Item relieflens_avg_severity.csv
```

Open it or import it with:

```powershell
python -c "from relieflens.exports.csv_io import import_csv; print(import_csv('relieflens_avg_severity.csv'))"
```

### 4.5 Run the built-in example script

A simple demo script is provided at `relieflens/example.py`.
Run it with:

```powershell
python relieflens/example.py
```

This executes the full new analytics pipeline and writes `relieflens_avg_severity.csv`.

### 4.6 Verify Cassandra simulation

```powershell
python -c "from relieflens.storage.cassandra_sim import CassandraSim, CQL_SCHEMA; print(CQL_SCHEMA); db=CassandraSim(); db.insert_batch([{'region':'ZoneA','disaster_type':'Flood','avg_severity':2.5,'timestamp':'now'}]); print(db.query_all())"
```

### 4.6 Verify Mongo simulation

```powershell
python -c "from relieflens.storage.mongo_sim import MongoSim; db=MongoSim(); db.insert('predictions', {'region':'ZoneA'}); db.increment_counter('counters','ZoneA',1); print(db.count_total('predictions')); print(db.count_by_field('predictions','region'))"
```

---

## 5. End-to-end run order for the full project

Use this order when running the ReliefLens project in full:

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python -m pip install -r requirement.txt`
4. `python src/preprocessing/build_dataset.py`
5. `python src/preprocessing/prepare_data.py`
6. `python src/preprocessing/clean_data.py --data-dir data/processed/final --apply` (optional)
7. `python run.py cnn_v2` or `python run.py cnn` or `python run.py svm`
8. `python run.py ui`
9. Run the new analytics test with `relieflens.pipeline`
10. Run the demo example script: `python relieflens/example.py`

---

## 6. What each added script teaches

### `mapreduce.py`

Teaches:

- MapReduce key-value mapping
- data shuffle/grouping
- reduce aggregation
- algorithmic complexity analysis

### `spark_sim.py`

Teaches:

- `flatMap` semantics
- map/reduceByKey pattern
- threshold filtering in analytics
- how Spark-style logic can be simulated locally

### `topk.py`

Teaches:

- hotspot detection
- Top-K ranking
- risk-based prioritisation in disaster analytics

### `threshold.py`

Teaches:

- decision thresholding
- how severity scoring can be built from probability outputs
- operational filtering for emergency triage

### `cassandra_sim.py`

Teaches:

- wide-column storage concepts
- CQL schema design
- batch insert, update, TTL, and schema evolution
- why TTL is important for temporary monitoring alerts

### `mongo_sim.py`

Teaches:

- document collection modeling
- atomic counter updates
- query-like counts and aggregation by field

### `csv_io.py`

Teaches:

- lightweight export/import for analytics
- interoperability between Python and tabular tools

### `pipeline.py`

Teaches:

- integrating multiple analytics components
- how a full pipeline can combine ML output, aggregation, monitoring, and export
- turning model predictions into actionable analytics results

---

## 7. Key presentation points for a Big Data analytics course

- The new code is not Hadoop or Spark, but it demonstrates the same architectural concepts locally.
- You can explain MapReduce with actual Python code and show the same complexity analysis used in distributed computing.
- The Spark simulation shows a practical `flatMap` + `reduceByKey` pipeline for log analytics.
- The Cassandra simulation uses real CQL text and shows batch insert, update, TTL, and schema extension.
- The Mongo simulation models collection inserts, counter updates, and field aggregation.
- This package layer is a bridge between ML model development and Big Data analytics concepts.

---

## 8. Summary of the new analytics layer

This project now contains two complementary parts:

1. **Core ReliefLens ML pipeline**
   - Data preparation
   - Model training
   - Inference UI

2. **New Big Data analytics simulation layer**
   - MapReduce aggregation
   - Spark-style log processing
   - Top-K hotspot detection
   - Threshold monitoring
   - Cassandra/Mongo-style storage simulation
   - CSV exporting for analytics

Together, these make ReliefLens both a machine learning solution and a Big Data analytics teaching example.
