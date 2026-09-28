"""
08_window_features.py - Medallion Silver Zone Window Analytical Feature Engineering Engine

Description:
    Computes rolling temporal window aggregation features over the enriched 31.8M+ transaction stream.
    Calculates a 3-transaction moving average of transfer amounts per account and tags monetary anomalies 
    (`is_amount_spike`) when a transaction payment significantly exceeds historical baseline metrics.

Architecture & Performance Optimizations:
    - Bounded Row Frame Execution: Defines explicit physical frame boundaries (`rowsBetween(-2, 0)`) 
      rather than unbounded range frames, preventing expensive whole-partition materialization and JVM memory bloat.
    - Composite Partitioning Strategy: Groups transaction history by composite key (`From Bank`, `Account`) 
      to distribute window aggregation processing evenly across Spark executor tasks.
    - Vectorized Catalyst Expression Evaluation: Derives spike flags using native `F.when()` conditional logic, 
      compiling analytical threshold checks directly into JVM bytecode without Python execution penalty.
    - Coalesced Parquet Storage: Enforces `.repartition(8)` prior to write ops to generate optimized Parquet file segments 
      and prevent small-file overhead in downstream graph aggregation pipelines.
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


def compute_window_features():
    """
    Computes 3-transaction rolling moving averages and amount spike indicators per account entity.
    """
    print("=== Executing Task 3.2: Window Analytical Feature & Amount Spike Pipeline ===")
    
    # Initialize containerized SparkSession instance configured with local execution limits
    spark = get_spark_session(app_name="AML_Silver_Window_Features", env="LOCAL")
    
    # Tune shuffle partitions to 8 to balance memory overhead during partition sorting
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    
    # Resolve Silver layer input and output directory paths
    input_path = os.path.join(project_root, "data", "silver", "enriched_transactions_parquet")
    output_path = os.path.join(project_root, "data", "silver", "transactions_window_features_parquet")
    
    # Read enriched Silver transaction DataFrame from Parquet storage
    print(f"[INFO] Reading enriched transactions dataset from: {input_path}")
    df = spark.read.parquet(input_path)
    
    # Define Window specification: Partition by account entity (Bank ID + Account Number), order by Timestamp,
    # and restrict execution frame explicitly to the current and previous 2 rows (rowsBetween(-2, 0)) to avoid unbounded JVM memory allocation
    window_spec = Window.partitionBy("From Bank", "Account").orderBy("Timestamp").rowsBetween(-2, 0)
    
    print("[INFO] Computing 3-row rolling moving average and detecting monetary spikes (is_amount_spike)...")
    
    # Compute rolling window average over normalized USD transfer amount
    moving_avg_col = F.avg(F.col("`Amount Paid USD`")).over(window_spec)
    
    # Append moving average feature and flag anomalous transactions exceeding 3x moving average baseline
    df_features = df.withColumn("moving_avg_amount_usd", moving_avg_col) \
                    .withColumn(
                        "is_amount_spike",
                        F.when(F.col("`Amount Paid USD`") > (F.col("moving_avg_amount_usd") * 3.0), 1).otherwise(0)
                    )
    
    # Write feature-engineered DataFrame back to Silver storage as repartitioned Parquet files
    print(f"[INFO] Persisting window feature dataset (8 partitions) to: {output_path}")
    df_features.repartition(8).write.mode("overwrite").parquet(output_path)
    
    print("[OK] Task 3.2 Analytical Window Feature Engineering successfully completed!")


if __name__ == "__main__":
    compute_window_features()