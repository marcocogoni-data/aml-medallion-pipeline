"""
10_temporal_cycle_patterns.py - Medallion Silver Zone Temporal Velocity & Circular Transfer Detection Engine

Description:
    Analyzes temporal execution dynamics and transactional graph cycles over the 31.8M+ record stream.
    Identifies two critical Anti-Money Laundering (AML) risk indicators:
    1. Rapid Pass-Through (High Velocity): Detects consecutive transactions on the same account 
       occurring within a 60-minute window.
    2. Circular Transfers (A -> B -> A Cycles): Detects round-trip laundering schemes between entity 
       pairs occurring within a 24-hour threshold.

Architecture & Performance Optimizations:
    - Native C-Level Unix Epoch Parsing: Converts string timestamps into 64-bit integer Unix epochs 
      (`unix_timestamp`), avoiding Python UDF overhead and enabling microsecond-precision subtraction arithmetic.
    - Undirected Pair Order Normalization: Employs `F.least()` and `F.greatest()` to construct symmetrical 
      entity pair keys (`pair_min`, `pair_max`), mapping directed transfers (A -> B and B -> A) into 
      a single stateful window partition without expensive network graph self-joins.
    - Stateful Sequence Tracking via Lag Windowing: Computes direction changes (`FORWARD` vs `REVERSE`) 
      and time deltas using `F.lag()`, evaluating temporal cycle velocity in \\(O(N \log N)\\) sort time.
    - JVM Intermediate State Garbage Collection: Explicitly drops intermediate calculation columns 
      prior to storage serialization to reduce Parquet block size and minimize JVM heap memory pressure.
"""

import os
import sys
import importlib
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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


def extract_temporal_cycle_patterns():
    """
    Computes rapid pass-through velocity and circular transfer round-trips across enriched transaction logs.
    """
    print("=== Executing Task 3.4: Temporal Velocity & Circular Transfer Detection Engine ===")
    
    # Initialize containerized SparkSession instance configured for local execution
    spark = get_spark_session(app_name="AML_Silver_Temporal_Cycle_Patterns", env="LOCAL")
    
    # Tune shuffle partitions and parallelism to 8 to optimize thread-level sorting on VM executors
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    spark.conf.set("spark.default.parallelism", "8")

    # Resolve input and output storage locations in the Silver Layer Medallion architecture
    input_path = os.path.join(project_root, "data", "silver", "transactions_window_features_parquet")
    output_path = os.path.join(project_root, "data", "silver", "temporal_cycle_patterns_parquet")

    # Read window-feature enriched Silver transactions dataset from Parquet storage
    print(f"[INFO] Reading window-feature enriched dataset from: {input_path}")
    df = spark.read.parquet(input_path)

    # -------------------------------------------------------------------------
    # 1. Robust Timestamp Parsing & Unix Epoch Standardization
    # Evaluates multiple timestamp formats into 64-bit integer Unix seconds to enable high-speed math
    # -------------------------------------------------------------------------
    print("[INFO] Parsing raw timestamps into normalized Unix Epoch seconds...")
    ts_col = F.col("Timestamp")
    unix_ts_expr = F.coalesce(
        ts_col.cast("long"),
        F.to_timestamp(ts_col, "yyyy/MM/dd HH:mm").cast("long"),
        F.to_timestamp(ts_col, "yyyy-MM-dd HH:mm:ss").cast("long"),
        F.unix_timestamp(ts_col)
    )
    df_prep = df.withColumn("unix_timestamp", unix_ts_expr)

    # -------------------------------------------------------------------------
    # 2. Rapid Pass-Through Velocity Detection (Intra-Account Inflow/Outflow <= 60 minutes)
    # -------------------------------------------------------------------------
    print("[INFO] Computing account-level transaction velocity (Rapid Pass-Through <= 60 min)...")
    
    # Window specification: partition by unique account key (Bank ID + Account) ordered chronologically
    window_account = Window.partitionBy("From Bank", "Account").orderBy("unix_timestamp")

    # Calculate time delta between consecutive transfers on the same account
    df_velocity = df_prep.withColumn(
        "prev_ts_account",
        F.lag(F.col("unix_timestamp"), 1).over(window_account)
    ).withColumn(
        "time_diff_minutes",
        F.when(
            F.col("prev_ts_account").isNotNull(),
            (F.col("unix_timestamp") - F.col("prev_ts_account")) / 60.0
        ).otherwise(None)
    ).withColumn(
        "is_rapid_passthrough",
        F.when((F.col("time_diff_minutes").isNotNull()) & (F.col("time_diff_minutes") <= 60), 1).otherwise(0)
    ).drop("prev_ts_account")

    # -------------------------------------------------------------------------
    # 3. Circular Transfer / Round-Trip Detection (Cycles A -> B -> A <= 24 hours)
    # -------------------------------------------------------------------------
    print("[INFO] Detecting circular entity transfers (Cycles A -> B -> A <= 24 hours)...")
    
    # Construct an undirected entity pair key (pair_min, pair_max) to bring both A->B and B->A into the same window partition
    df_cycles_prep = df_velocity.withColumn(
        "pair_min", F.least(F.col("from_entity_id"), F.col("to_entity_id"))
    ).withColumn(
        "pair_max", F.greatest(F.col("from_entity_id"), F.col("to_entity_id"))
    ).withColumn(
        "transfer_dir",
        F.when(F.col("from_entity_id") == F.col("pair_min"), F.lit("FORWARD")).otherwise(F.lit("REVERSE"))
    )

    # Window specification: partition by entity pair, ordered by execution timestamp
    window_pair = Window.partitionBy("pair_min", "pair_max").orderBy("unix_timestamp")

    # Evaluate direction reversal and time delta between back-and-forth transfers
    df_final = df_cycles_prep.withColumn(
        "prev_dir", F.lag(F.col("transfer_dir"), 1).over(window_pair)
    ).withColumn(
        "prev_ts_pair", F.lag(F.col("unix_timestamp"), 1).over(window_pair)
    ).withColumn(
        "cycle_diff_hours",
        F.when(
            (F.col("prev_dir").isNotNull()) & (F.col("transfer_dir") != F.col("prev_dir")),
            (F.col("unix_timestamp") - F.col("prev_ts_pair")) / 3600.0
        ).otherwise(None)
    ).withColumn(
        "is_circular_transfer",
        F.when((F.col("cycle_diff_hours").isNotNull()) & (F.col("cycle_diff_hours") <= 24), 1).otherwise(0)
    ).drop(
        "unix_timestamp", "pair_min", "pair_max", "transfer_dir", 
        "prev_dir", "prev_ts_pair", "cycle_diff_hours"
    )

    # -------------------------------------------------------------------------
    # 4. Partition Rebalancing & Output Export
    # -------------------------------------------------------------------------
    print(f"[INFO] Persisting temporal and cycle pattern dataset (8 partitions) to: {output_path}")
    df_final.repartition(8).write.mode("overwrite").parquet(output_path)

    print("[OK] Task 3.4 Temporal & Cycle Pattern Extraction successfully completed!")


if __name__ == "__main__":
    extract_temporal_cycle_patterns()