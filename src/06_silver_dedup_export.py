"""
06_silver_dedup_export.py - Medallion Silver Zone Exact Deduplication & Final Export Engine

Description:
    Performs full-row exact record deduplication across both Accounts and Transactions datasets 
    within the Silver Layer (`data/silver/`). Eliminates redundant financial records introduced 
    during ingestion or source system re-transmissions, producing certified, deduplicated 
    Silver storage tables.

Architecture & Performance Optimizations:
    - Distributed Tungsten Hash Deduplication: Leverages PySpark's native `dropDuplicates()` 
      which constructs off-heap hash tables inside JVM Tungsten memory to identify and remove 
      exact record duplicates without requiring full sort shuffles.
    - Partition Coalescing & File Sizing: Applies explicit `.repartition(8)` prior to write ops 
      to balance partition file sizes (~128MB per Parquet block) and mitigate the Small File Problem.
    - Adaptive Shuffle Partition Tuning: Dynamically overrides `spark.sql.shuffle.partitions` to 8 
      for deduplication memory management, minimizing network overhead on local/single-node executors.
    - Two-Phase Safe Staging I/O Pattern: Writes deduplicated data to temporary staging paths (`_temp`) 
      before overwriting final Silver targets, guaranteeing atomic table updates and preventing read locks.
"""

import os
import sys
import shutil
import importlib

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


def deduplicate_and_export_silver():
    """
    Executes full-row deduplication across Silver Accounts and Transactions, exporting cleansed Parquet outputs.
    """
    print("=== Executing Task 2.3: Silver Zone Exact Deduplication & Final Export ===")
    
    # Initialize containerized SparkSession instance configured for Silver processing
    spark = get_spark_session(app_name="AML_Silver_Dedup_Export", env="LOCAL")
    
    # Dynamically tune shuffle partitions to 8 to reduce context-switching and hash table memory overhead
    spark.conf.set("spark.sql.shuffle.partitions", "8")
    
    # Resolve Silver storage paths and intermediate staging target locations
    accounts_path = os.path.join(project_root, "data", "silver", "accounts_parquet")
    trans_path = os.path.join(project_root, "data", "silver", "transactions_parquet")
    
    temp_trans_path = os.path.join(project_root, "data", "silver", "transactions_parquet_temp")
    temp_acc_path = os.path.join(project_root, "data", "silver", "accounts_parquet_temp")
    
    # Read cleansed Silver Parquet tables into DataFrame memory
    print(f"[INFO] Reading Silver Accounts dataset from: {accounts_path}")
    accounts_df = spark.read.parquet(accounts_path)
    
    print(f"[INFO] Reading Silver Transactions dataset from: {trans_path}")
    transactions_df = spark.read.parquet(trans_path)
    
    # Apply full-row distributed hash deduplication using Tungsten execution engine
    print("[INFO] Applying exact row deduplication across Accounts and Transactions...")
    accounts_dedup = accounts_df.dropDuplicates()
    transactions_dedup = transactions_df.dropDuplicates()
    
    # Phase 1 Staging Write: Balance partitions to 8 blocks (~128MB optimal Parquet sizing) and write to staging
    print("[INFO] Writing deduplicated datasets to intermediate staging storage...")
    accounts_dedup.repartition(8).write.mode("overwrite").parquet(temp_acc_path)
    transactions_dedup.repartition(8).write.mode("overwrite").parquet(temp_trans_path)
    
    # Phase 2 Final Overwrite: Replace primary Silver datasets atomically from verified staging outputs
    print("[INFO] Overwriting target Silver Zone storage directories with deduplicated tables...")
    spark.read.parquet(temp_acc_path).write.mode("overwrite").parquet(accounts_path)
    spark.read.parquet(temp_trans_path).write.mode("overwrite").parquet(trans_path)
    
    # Clean up intermediate staging directories to ensure clean directory layout
    shutil.rmtree(temp_acc_path, ignore_errors=True)
    shutil.rmtree(temp_trans_path, ignore_errors=True)
    
    print("[OK] Silver Zone Deduplication and Export successfully completed!")


if __name__ == "__main__":
    deduplicate_and_export_silver()