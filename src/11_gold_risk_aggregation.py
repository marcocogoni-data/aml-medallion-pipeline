"""
11_gold_risk_aggregation.py - Medallion Gold Zone Synthetic AML Risk Scoring & Consolidation Engine

Description:
    Consolidates temporal velocity flags, amount spike anomalies, and network graph topology metrics 
    from the Silver Layer to compute a normalized, composite Anti-Money Laundering (AML) Risk Score (0-100) 
    per account entity. Exports certified risk-scored alerts into the Gold Layer (`data/gold/risk_score_aggregation_parquet`).

Architecture & Performance Optimizations:
    - Pure Vectorized Catalyst Risk Expression: Avoids Python `@udf` execution overhead by utilizing native 
      PySpark SQL arithmetic expressions and `F.least(F.lit(100.0), ...)` for deterministic score capping.
    - Optimized Left Join Topology Resolution: Selects and deduplicates target topology entities (`dropDuplicates`) 
      prior to joining, preventing row multiplication during Silver-to-Gold aggregation.
    - Null-Safe Coalesce Pattern: Replaces missing graph topology join attributes with default zero values (`F.coalesce`), 
      guaranteeing numerical integrity during weighted score summation.
    - Coalesced Storage Allocation: Writes final Gold risk tables using explicit `.repartition(8)` to optimize 
      Parquet file sizes (~128MB) for downstream Business Intelligence (BI) and compliance audit tools.
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


def aggregate_gold_risk_score():
    """
    Consolidates Silver layer risk indicators and graph topologies into a unified Gold Zone AML risk score (0-100).
    """
    print("=== Executing Task 4.1: Gold Zone Risk Score Consolidation & Aggregation ===")
    
    # Initialize containerized SparkSession instance configured for Gold layer processing
    spark = get_spark_session(app_name="AML_Gold_Risk_Aggregation", env="LOCAL")
    
    # Tune shuffle partitions to 8 to balance memory overhead during final entity joins
    spark.conf.set("spark.sql.shuffle.partitions", "8")

    # Resolve input Silver paths and target Gold storage location
    temporal_path = os.path.join(project_root, "data", "silver", "temporal_cycle_patterns_parquet")
    topology_path = os.path.join(project_root, "data", "silver", "topology_patterns_parquet")
    output_path = os.path.join(project_root, "data", "gold", "risk_score_aggregation_parquet")

    # Read temporal/cycle feature enriched dataset from Silver Parquet storage
    print(f"[INFO] Reading Silver temporal & cycle pattern dataset from: {temporal_path}")
    transactions_df = spark.read.parquet(temporal_path)

    # Read topological fraud pattern dataset from Silver Parquet storage
    print(f"[INFO] Reading Silver topology pattern dataset from: {topology_path}")
    topology_data_df = spark.read.parquet(topology_path)

    # -------------------------------------------------------------------------
    # 1. Entity & Account Level Risk Indicator Aggregation
    # Computes maximum boolean anomaly flags per entity/account to capture worst-case risk state
    # -------------------------------------------------------------------------
    print("[INFO] Aggregating transaction risk flags per originating entity and account...")
    aggregated_transactions_df = transactions_df.groupBy(
        "from_entity_id", "from_entity_name", "From Bank", "Account"
    ).agg(
        F.max("is_amount_spike").alias("is_amount_spike_max"),
        F.max("is_rapid_passthrough").alias("is_rapid_passthrough_max"),
        F.max("is_circular_transfer").alias("is_circular_transfer_max")
    ).withColumnRenamed("from_entity_id", "entity_id")

    # -------------------------------------------------------------------------
    # 2. Map-Side Entity Join with Topological Patterns
    # Prunes and deduplicates topology entities to prevent row explosion during Left Outer Join
    # -------------------------------------------------------------------------
    print("[INFO] Joining entity aggregations with graph topology risk patterns...")
    topology_pruned = topology_data_df.select(
        F.col("entity_id").alias("top_entity_id"),
        F.col("degree_count"),
        F.col("pattern_type")
    ).dropDuplicates(["top_entity_id"])

    joined_df = aggregated_transactions_df.join(
        topology_pruned,
        aggregated_transactions_df["entity_id"] == topology_pruned["top_entity_id"],
        how="left"
    ).drop("top_entity_id")

    # -------------------------------------------------------------------------
    # 3. Null Imputation & Topology Risk Presence Flagging
    # Safely handles non-matched entities using F.coalesce to replace Null degree counts with 0
    # -------------------------------------------------------------------------
    joined_df = joined_df.withColumn(
        "degree_count", F.coalesce(F.col("degree_count"), F.lit(0))
    ).withColumn(
        "has_topology_pattern",
        F.when(F.col("degree_count") > 0, 1).otherwise(0)
    )

    # -------------------------------------------------------------------------
    # 4. Composite AML Risk Score Calculation (Scale: 0 to 100)
    # Applies weighted risk formula natively in Catalyst memory space without Python UDF overhead:
    # - Amount Spike = 25 pts | Rapid Pass-Through = 30 pts | Circular Transfer = 15 pts | Topology = 30 pts
    # -------------------------------------------------------------------------
    print("[INFO] Computing composite weighted AML risk score (0-100 scale)...")
    raw_score_expr = (
        F.col("is_amount_spike_max") * 25 +
        F.col("is_rapid_passthrough_max") * 30 +
        F.col("is_circular_transfer_max") * 15 +
        F.col("has_topology_pattern") * 30
    )

    # Calculate raw score sum and cap final risk score strictly at 100.0
    gold_df = joined_df.withColumn(
        "raw_score", raw_score_expr
    ).withColumn(
        "risk_score",
        F.least(F.lit(100.0), F.col("raw_score").cast("double"))
    )

    # -------------------------------------------------------------------------
    # 5. Partition Rebalancing & Gold Zone Export
    # -------------------------------------------------------------------------
    print(f"[INFO] Persisting certified Gold risk-scored dataset (8 partitions) to: {output_path}")
    gold_df.repartition(8).write.mode("overwrite").parquet(output_path)

    print("[OK] Task 4.1 Gold Zone Risk Score Aggregation successfully completed!")


if __name__ == "__main__":
    aggregate_gold_risk_score()