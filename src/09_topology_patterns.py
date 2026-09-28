"""
09_topology_patterns.py - Medallion Silver Zone Graph Topology & Network Fraud Pattern Engine

Description:
    Detects structural graph topologies associated with Anti-Money Laundering (AML) fraud schemes, 
    specifically Fan-In aggregation (many-to-one account pooling) and Fan-Out dispersion/smurfing 
    (one-to-many structuring). Evaluates directional entity degree metrics (In-Degree and Out-Degree > 5) 
    and outputs a unified topological risk dataset into the Silver Layer (`data/silver/topology_patterns_parquet`).

Architecture & Performance Optimizations:
    - Directional Graph Degree Aggregations: Leverages `F.countDistinct()` within distributed `groupBy` 
      execution pipelines to evaluate entity-level network degree metrics across 31.8M+ transactions.
    - Push-Down Threshold Filtering: Applies immediate post-aggregation boolean predicates (`degree_count > 5`) 
      to prune low-risk entities early, drastically reducing intermediate state memory overhead.
    - Zero-Shuffle Schema Unification: Standardizes projection aliases (`entity_id`, `entity_name`, `pattern_type`) 
      across Fan-In and Fan-Out queries to enable seamless, zero-shuffle vertical concatenation via `union()`.
    - Coalesced Storage Allocation: Writes result sets back to Parquet format using explicit `.repartition(8)` 
      to eliminate small-file fragmentation prior to Gold layer risk aggregation.
"""

import os
import sys
import importlib
from pyspark.sql import functions as F

# Inject Hadoop native binary dependencies (WinUtils) explicitly into the OS PATH environment
# to ensure local storage I/O compatibility when executing on Windows OS environments
os.environ["HADOOP_HOME"] = "C:\\hadoop"
hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
if hadoop_bin not in os.environ["PATH"]:
    os.environ["PATH"] = hadoop_bin + ";" + os.environ["PATH"]

# Dynamically resolve workspace execution root and script directory boundaries
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Ensure current module directory and project root are prioritized in Python sys.path lookup table
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dynamically load the centralized SparkSession factory using importlib to bypass numeric naming restrictions
spark_mod = importlib.import_module("01_spark_session")
get_spark_session = spark_mod.get_spark_session


def extract_topology_patterns():
    """
    Identifies structural graph anomalies (Fan-In pooling and Fan-Out dispersion) across enriched entities.
    """
    print("=== Executing Task 3.3: Graph Topology Pattern Detection (Fan-In / Fan-Out) ===")
    
    # Initialize containerized SparkSession instance configured with local execution parameters
    spark = get_spark_session(app_name="AML_Silver_Topology_Patterns", env="LOCAL")
    
    # Tune shuffle partitions to 8 to balance memory overhead during entity aggregation shuffles
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    
    # Resolve input and output directory targets within Silver Layer Medallion architecture
    input_path = os.path.join(project_root, "data", "silver", "transactions_window_features_parquet")
    output_path = os.path.join(project_root, "data", "silver", "topology_patterns_parquet")
    
    # Read window-feature enriched transactions DataFrame from Silver Parquet storage
    print(f"[INFO] Reading window-feature enriched dataset from: {input_path}")
    df = spark.read.parquet(input_path)
    
    # -------------------------------------------------------------------------
    # 1. Fan-In Aggregation (In-Degree: Multiple Sender Entities -> Single Target Entity)
    # Identifies central destination entities receiving funds from > 5 distinct originating entities
    # -------------------------------------------------------------------------
    print("[INFO] Aggregating Fan-In graph topology metrics (In-Degree > 5)...")
    fan_in_df = df.groupBy("to_entity_id", "to_entity_name").agg(
        F.countDistinct("from_entity_id").alias("degree_count"),
        F.sum(F.col("`Amount Paid USD`")).alias("total_amount_usd")
    ).filter(F.col("degree_count") > 5) \
     .withColumn("pattern_type", F.lit("FAN_IN")) \
     .select(
        F.col("to_entity_id").alias("entity_id"),
        F.col("to_entity_name").alias("entity_name"),
        F.col("pattern_type"),
        F.col("degree_count"),
        F.col("total_amount_usd")
    )
    
    # -------------------------------------------------------------------------
    # 2. Fan-Out / Smurfing Aggregation (Out-Degree: Single Source Entity -> Multiple Receiver Entities)
    # Identifies central source entities dispersing funds out to > 5 distinct destination entities
    # -------------------------------------------------------------------------
    print("[INFO] Aggregating Fan-Out / Smurfing graph topology metrics (Out-Degree > 5)...")
    fan_out_df = df.groupBy("from_entity_id", "from_entity_name").agg(
        F.countDistinct("to_entity_id").alias("degree_count"),
        F.sum(F.col("`Amount Paid USD`")).alias("total_amount_usd")
    ).filter(F.col("degree_count") > 5) \
     .withColumn("pattern_type", F.lit("FAN_OUT")) \
     .select(
        F.col("from_entity_id").alias("entity_id"),
        F.col("from_entity_name").alias("entity_name"),
        F.col("pattern_type"),
        F.col("degree_count"),
        F.col("total_amount_usd")
    )
    
    # -------------------------------------------------------------------------
    # 3. Relational Schema Unification & Result Merging
    # Concatenates Fan-In and Fan-Out DataFrames into a standardized topological risk dataset
    # -------------------------------------------------------------------------
    print("[INFO] Merging Fan-In and Fan-Out topology datasets...")
    topology_df = fan_in_df.union(fan_out_df)
    
    # -------------------------------------------------------------------------
    # 4. Partition Coalescing & Output Export
    # -------------------------------------------------------------------------
    print(f"[INFO] Persisting topological risk patterns (8 partitions) to: {output_path}")
    topology_df.repartition(8).write.mode("overwrite").parquet(output_path)
    
    print("[OK] Task 3.3 Graph Topology Extraction successfully completed!")


if __name__ == "__main__":
    extract_topology_patterns()