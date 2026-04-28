# 1. INTRODUCTION

Natural disasters such as floods, earthquakes, cyclones, and wildfires produce large volumes of heterogeneous data in the form of aerial, satellite, and drone imagery. The ability to process and analyse such large-scale data efficiently is critical for disaster response, resource allocation, and risk assessment. However, traditional manual approaches to analysing disaster data are time-consuming, labour-intensive, and incapable of scaling to the size and speed at which data is generated.

This project presents ReliefLens, a system designed to apply Big Data Analytics techniques for processing and analysing disaster-related data. The primary focus of the system is not on raw image processing, but on analysing structured outputs derived from large datasets to extract meaningful insights such as severity distribution, regional impact, and hotspot identification.

The system incorporates key Big Data concepts such as:

MapReduce-based batch aggregation
Spark-style transformation pipelines
Distributed storage models
Integrated analytics pipelines

These concepts are implemented using simulation-based approaches to demonstrate how large-scale data processing systems function in real-world environments.

The objectives of the project are:

To integrate large-scale heterogeneous datasets into a unified format
To clean and preprocess data to improve reliability and consistency
To apply scalable analytics techniques for extracting patterns
To identify high-risk regions using ranking and aggregation methods
To demonstrate Big Data processing paradigms using simulation

The project highlights the importance of transforming raw data into actionable insights using structured analytics, which is a core principle of Big Data Analytics.

# 2. DATA COLLECTION AND INSPECTION

## 2.1 Data Sources

The dataset used in this project is constructed by combining multiple sources of disaster-related imagery. These include aerial survey datasets, drone-captured images, and publicly available satellite imagery. Each dataset contributes unique characteristics in terms of disaster type, resolution, and capture conditions. :contentReference[oaicite:0]{index=0}

## 2.2 Nature of the Data

The collected dataset demonstrates the fundamental characteristics of Big Data:

Volume

The dataset consists of tens of thousands of images, making it large enough to require efficient storage and processing techniques.

Variety

The data varies significantly across several dimensions:

Image formats and resolutions
Lighting conditions and visibility
Types of disasters represented
Capture perspectives and angles

Velocity

In real-world scenarios, disaster-related data is generated rapidly, especially during active events where drones and satellites continuously capture images.

Veracity

The dataset contains noise and inconsistencies, including:

Blurry or low-quality images
Duplicate or near-duplicate frames
Inconsistent metadata

## 2.3 Data Inspection Process

The dataset was systematically inspected to identify quality issues and structural inconsistencies. The following problems were observed:

Redundant images due to continuous capture in drone footage
Presence of images with insufficient detail or clarity
Uneven distribution of data across regions and disaster types
Lack of uniform structure across different data sources

These issues highlighted the need for a structured data cleaning and preprocessing pipeline.

## 2.4 Data Cleaning Techniques

Duplicate Detection and Removal

Perceptual hashing techniques were used to identify duplicate and near-duplicate images. Each image was converted into a compact hash representation, and images with similar hashes were grouped together. Only one representative image from each group was retained.

This step reduces redundancy and ensures that analytical results are not biased by repeated data.

Blur Detection

Blur detection was performed using Laplacian variance. Images with variance below a predefined threshold were considered blurry and removed from the dataset.

Blurry images lack useful information and negatively impact analytical accuracy.

Data Backup Strategy

Instead of permanently deleting unwanted data, all removed images were stored in a backup directory. This allows recovery of data if cleaning parameters need to be adjusted later.

## 2.5 Data Transformation

After cleaning, the dataset was transformed into a structured format suitable for analytics:

{region, disaster_type, predicted_class, probabilities, timestamp}

This structured representation enables efficient batch processing, aggregation, and analysis.

## 2.6 Key Observations from Inspection

Data distribution is highly uneven across regions
Severe cases are relatively fewer but critically important
Duplicate-heavy data can distort aggregation results
Data variability reflects real-world conditions

These observations influenced the design of the analytics pipeline.

# 3. BIG DATA APPLICATION

## 3.1 MapReduce-Based Aggregation

MapReduce is used to perform large-scale aggregation of severity data.

Map Phase

Each input record is transformed into a key-value pair:

((region, disaster_type), severity_score)

This step distributes data into intermediate key-value representations.

Shuffle and Sort Phase

All key-value pairs are grouped based on their keys:

(region, disaster_type) → [s1, s2, ..., sn]

This step simulates data redistribution across nodes in a distributed system.

Reduce Phase

For each group, the average severity is computed:

avg_severity = (Σ severity_scores) / n

Complexity Analysis

Map Phase: O(N)
Shuffle Phase: O(N log N)
Reduce Phase: O(N)

Overall complexity: O(N log N)

Significance

Reduces large datasets into aggregated metrics
Enables region-wise and disaster-wise comparison
Improves interpretability of data

## 3.2 Spark-Style Data Processing

A Spark-inspired pipeline is implemented to process log data.

Processing Steps

flatMap: Extracts all error entries from logs
map: Converts each error into a key-value pair
reduceByKey: Aggregates counts for each error
filter: Retains errors above a frequency threshold

Purpose

Identifies recurring issues in the system
Detects anomalies in processing
Supports monitoring and debugging

## 3.3 Top-K Hotspot Detection

Top-K analysis is used to identify regions with the highest number of severe cases.

Algorithm Steps

Filter records with severity above a threshold
Count occurrences per region
Sort regions in descending order
Select top K regions

Complexity

Counting: O(N)
Sorting: O(N log N)

Outcome

Identification of high-risk regions
Supports prioritised resource allocation

## 3.4 Threshold-Based Filtering

A threshold-based approach is used to classify critical cases.

Definition

severity_score = p(major) + p(destroyed)

If:

severity_score ≥ threshold → critical case

Role

Simplifies decision-making
Converts probability-based data into actionable categories
Enables fast filtering of large datasets

## 3.5 Distributed Storage Simulation

Cassandra-Like Storage

Uses wide-column data model
Supports TTL for temporary data
Efficient for large-scale writes

MongoDB-Like Storage

Uses document-based structure
Supports flexible schema
Enables aggregation queries

Purpose

Demonstrates how large-scale systems store analytics results
Simulates distributed storage behaviour

## 3.6 Data Export

CSV export functionality is implemented to:

Store aggregated results
Enable external analysis
Support reporting and visualisation

## 3.7 Integrated Analytics Pipeline

The pipeline integrates all components:

Data input
Aggregation (MapReduce)
Hotspot detection (Top-K)
Threshold filtering
Log analysis (Spark-style)
Storage
Export

This pipeline converts raw data into structured insights.

# 4. EXPLORING DATA DISTRIBUTION AND ANALYSIS

## 4.1 Distribution Analysis

The dataset shows:

Skewed distribution across regions
Imbalance between severity levels
Variability across disaster types

## 4.2 Statistical Insights

Mean severity differs significantly across regions
Variance indicates instability in impact levels
Distribution is non-uniform

## 4.3 Aggregation Insights

Aggregation enables:

Clear comparison across regions
Reduction of noise
Improved interpretability

## 4.4 Pattern Identification

Certain regions consistently show higher severity
Disaster types influence severity distribution
Recurring patterns indicate vulnerability

## 4.5 Analytical Observations

Aggregation transforms raw data into insights
Filtering isolates critical cases
Ranking enables prioritisation

# 5. RESULTS AND DISCUSSION

## 5.1 Aggregation Results

Average severity successfully computed per region
Significant variation observed across regions
High-risk regions identified

## 5.2 Hotspot Detection Results

Top-K regions identified as critical zones
High concentration of severe cases in selected regions
Enables prioritised response

## 5.3 Threshold Filtering Results

Critical cases effectively isolated
Reduced dataset complexity
Improved focus on high-risk data

## 5.4 Log Analysis Results

Frequent error patterns identified
System anomalies detected
Supports system monitoring

## 5.5 Discussion

Aggregated data provides better insights than raw data
Analytical techniques improve decision-making efficiency
Pipeline successfully handles large-scale data

## 5.6 Limitations

Simulation does not fully replicate distributed systems
Data quality impacts accuracy
Real-time processing requires additional infrastructure

# 6. CONCLUSION

This project demonstrates the effective application of Big Data Analytics techniques for analysing large-scale disaster data. By integrating MapReduce, Spark-style processing, distributed storage, and analytics pipelines, the system transforms raw data into actionable insights.

The analysis highlights the importance of aggregation, filtering, and ranking in extracting meaningful patterns from large datasets. The system successfully identifies high-risk regions and supports data-driven decision-making.

The project provides a practical implementation of Big Data concepts and can be extended by integrating real distributed systems and real-time processing frameworks for enhanced scalability and performance.
