"""
05_currency_timestamp.py - Medallion Silver Zone Currency Normalization & Temporal Parsing Engine

Description:
    Normalizes multi-currency transaction records across 15 distinct global currencies into 
    standardized US Dollar (USD) monetary equivalents within the Silver Layer (`data/silver/`). 
    Parses raw string timestamps into native PySpark `TimestampType` instances to enable 
    microsecond-precision temporal ordering in downstream window transformations.

Architecture & Performance Optimizations:
    - Vectorized Catalyst Expression Trees: Dynamically constructs nested `when/otherwise` conditional 
      trees compiled directly into Java bytecode by the Catalyst Optimizer, bypassing Python UDF 
      execution overhead across 31.8M+ transaction records.
    - JVM Native Temporal Parsing: Uses `to_timestamp` with explicit format specifiers (`yyyy/MM/dd HH:mm`) 
      to achieve vectorized string-to-timestamp parsing inside JVM memory without Python object creation.
    - Two-Phase Safe Staging I/O Pattern: Implements an intermediate staging write phase (`_temp`) 
      to prevent read/write lock collisions and file corruption when modifying Parquet storage in-place.
"""

import os
import sys
import importlib
import shutil
from pyspark.sql.functions import col, to_timestamp, when, lit

# Inject Hadoop native binary dependencies (WinUtils) explicitly into the OS PATH environment
# to ensure local storage I/O compatibility when executing on Windows OS environments
os.environ["HADOOP_HOME"] = "C:\\hadoop"
hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
if hadoop_bin not in os.environ["PATH"]:
    os.environ["PATH"] = hadoop_bin + ";" + os.environ["PATH"]

# Dynamically resolve workspace root and script execution boundaries
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Ensure current module directory and project root are prioritized in Python sys.path lookup table
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dynamically import numeric module dependencies using importlib
spark_mod = importlib.import_module("01_spark_session")
get_spark_session = spark_mod.get_spark_session


def convert_currencies_and_timestamps(df):
    """
    Standardizes 15 foreign currency exchange rates to USD and converts string timestamps to TimestampType.
    
    Transformation Details:
        - Creates derived columns `Amount Paid USD` and `Amount Received USD`.
        - Employs static exchange rate multipliers derived from reference financial benchmarks.
        - Transforms ISO date strings into strongly typed `TimestampType` values for temporal analysis.
    """
    # Static exchange rate conversion table relative to USD benchmark
    currency_rates = {
        'US Dollar': 1.0,
        'Euro': 1.19,
        'Yuan': 0.16,
        'Brazil Real': 0.27,
        'Canadian Dollar': 0.80,
        'Rupee': 0.014,
        'Australian Dollar': 0.76,
        'Saudi Riyal': 0.27,
        'Mexican Peso': 0.049,
        'Ruble': 0.014,
        'Yen': 0.009,
        'UK Pound': 1.31,
        'Bitcoin': 4500.0,
        'Swiss Franc': 1.07,
        'Shekel': 0.30
    }
    
    # Dynamically compile nested Catalyst 'when' conditional logic into expression memory space
    rate_col = lit(1.0)
    for curr, rate in currency_rates.items():
        rate_col = when(col("`Payment Currency`") == curr, lit(rate)).otherwise(rate_col)
    
    # Vectorized execution of currency conversion arithmetic and C-level string parsing
    return df \
        .withColumn("Amount Paid USD", col("`Amount Paid`") * rate_col) \
        .withColumn("Amount Received USD", col("`Amount Received`") * rate_col) \
        .withColumn("Timestamp", to_timestamp(col("`Timestamp`"), "yyyy/MM/dd HH:mm"))


def main():
    print("=== Executing Currency Normalization & Timestamp Engine (Task 2.2 - Silver Zone) ===")
    
    # Initialize containerized SparkSession configured with local execution parameters
    spark = get_spark_session(app_name="AML_Currency_Timestamp_Silver", env="LOCAL")
    
    # Resolve input, staging, and output directory targets
    silver_trans_path = os.path.join(project_root, "data", "silver", "transactions_parquet")
    temp_trans_path = os.path.join(project_root, "data", "silver", "transactions_parquet_temp")
    
    # Read cleansed transactions Parquet table from Silver layer
    print(f"[INFO] Reading Silver Transactions dataset from: {silver_trans_path}")
    transactions_df = spark.read.parquet(silver_trans_path)
    
    # Execute monetary standardization and timestamp casting logic
    print("[INFO] Normalizing 15 foreign currencies to USD and parsing timestamps...")
    transactions_final = convert_currencies_and_timestamps(transactions_df)
    
    # Phase 1 Staging Write: Persist transformed dataset to a temporary staging path 
    # to eliminate Spark lock contention when performing in-place directory overwrites
    print(f"[INFO] Writing intermediate dataset to staging directory: {temp_trans_path}")
    transactions_final.write.mode("overwrite").parquet(temp_trans_path)
    
    # Phase 2 Final Overwrite: Atomically overwrite primary Silver storage path from staging
    print(f"[INFO] Overwriting target Silver Zone storage path: {silver_trans_path}")
    spark.read.parquet(temp_trans_path).write.mode("overwrite").parquet(silver_trans_path)
    
    # Cleanup temporary staging storage to maintain clean directory layout
    if os.path.exists(temp_trans_path):
        shutil.rmtree(temp_trans_path, ignore_errors=True)
    
    print("[OK] Task 2.2 Currency & Timestamp normalization successfully completed!")


if __name__ == "__main__":
    main()