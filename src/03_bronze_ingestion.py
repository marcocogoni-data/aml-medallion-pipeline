"""
03_bronze_ingestion.py - Medallion Bronze Zone Raw Data Ingestion Engine

Description:
    Ingests raw unstructured and semi-structured banking datasets (CSV and TXT formats) from 
    the landing zone (`data/raw/`) into the Medallion Bronze Layer (`data/bronze/`). Converts 
    textual data into columnar Apache Parquet format with default Snappy compression.

Architecture & Performance Optimizations:
    - Columnar Parquet Serialization: Converts line-oriented CSVs into binary columnar Parquet storage, 
      enabling high-throughput row group skipping, columnar projection, and data compression.
    - Explicit Schema Enforcement: Integrates `02_schemas.py` StructTypes to bypass automatic CSV 
      schema inference, eliminating a full 31.8M row I/O scanning pass across raw storage.
    - Dynamic Module Resolution: Uses Python's `importlib` to bypass numeric module naming constraints 
      during runtime initialization.
    - I/O & Native Library Stability: Configures Spark Hadoop file system properties to override 
      Windows native C++ filesystem library dependencies (`LocalFileSystem`), preventing I/O runtime crashes.
"""

import sys
import os
import importlib

# Inject Hadoop native binary dependencies (WinUtils) explicitly into the OS PATH environment
# to ensure local storage I/O compatibility when executing on Windows operating systems
os.environ["HADOOP_HOME"] = "C:\\hadoop"
hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
if hadoop_bin not in os.environ["PATH"]:
    os.environ["PATH"] = hadoop_bin + ";" + os.environ["PATH"]

# Dynamically construct relative execution paths to support seamless local/VM directory resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Ensure current module directory and project root are prioritized in Python sys.path lookup table
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dynamically import numeric module dependencies to circumvent Python PEP-8 string identifier restrictions
spark_session_mod = importlib.import_module("01_spark_session")
schemas_mod = importlib.import_module("02_schemas")

get_spark_session = spark_session_mod.get_spark_session
get_accounts_schema = schemas_mod.get_accounts_schema
get_transactions_schema = schemas_mod.get_transactions_schema
get_patterns_schema = schemas_mod.get_patterns_schema


def bronze_ingestion():
    """
    Executes the ingestion pipeline converting raw CSV/TXT datasets into compressed Bronze Parquet tables.
    
    Returns:
        SparkSession: Active Spark session reference for downstream verification logging.
    """
    # Initialize local/containerized SparkSession instance optimized for local memory limits
    spark = get_spark_session(app_name="AML_Bronze_Ingestion", env="LOCAL")
    
    # Disable native Hadoop C++ library checks to ensure pure Java fallback on local file systems
    spark.conf.set("spark.hadoop.io.native.lib.available", "false")
    spark.conf.set("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
    
    # -------------------------------------------------------------------------
    # 1. Accounts Dataset Ingestion (2.1M Rows: Raw CSV -> Bronze Parquet)
    # -------------------------------------------------------------------------
    accounts_path = os.path.join(project_root, "data", "raw", "HI-Medium_accounts.csv")
    print(f"[INFO] Ingesting Accounts dataset from: {accounts_path}...")
    
    # Enforce explicit StructType schema to prevent schema inference CPU overhead
    accounts_df = spark.read.csv(accounts_path, header=True, schema=get_accounts_schema())
    accounts_output = os.path.join(project_root, "data", "bronze", "accounts_parquet")
    
    # Write to Bronze layer as columnar Parquet with default Snappy block compression
    accounts_df.write.mode("overwrite").parquet(accounts_output)
    
    # -------------------------------------------------------------------------
    # 2. Transactions Dataset Ingestion (31.8M Rows: Raw CSV -> Bronze Parquet)
    # -------------------------------------------------------------------------
    trans_path = os.path.join(project_root, "data", "raw", "HI-Medium_Trans.csv")
    print(f"[INFO] Ingesting Transactions dataset from: {trans_path}...")
    
    # Direct explicit ingestion over 31.8M rows; bypasses line-by-line type checking
    transactions_df = spark.read.csv(trans_path, header=True, schema=get_transactions_schema())
    trans_output = os.path.join(project_root, "data", "bronze", "transactions_parquet")
    
    # Write transactions to raw Bronze zone; establishes immutability baseline for downstream auditability
    transactions_df.write.mode("overwrite").parquet(trans_output)
    
    # -------------------------------------------------------------------------
    # 3. Laundering Patterns Dataset Ingestion (31K Rows: Raw Text -> Bronze Parquet)
    # -------------------------------------------------------------------------
    patterns_path = os.path.join(project_root, "data", "raw", "HI-Medium_Patterns.txt")
    print(f"[INFO] Ingesting Patterns dataset from: {patterns_path}...")
    
    # Read unstructured line text file and map value column explicitly for graph pattern processing
    patterns_df = spark.read.text(patterns_path).withColumnRenamed("value", "Pattern_Line")
    patterns_output = os.path.join(project_root, "data", "bronze", "patterns_parquet")
    
    # Persist serialized pattern lines into Bronze Parquet storage
    patterns_df.write.mode("overwrite").parquet(patterns_output)
    
    return spark


if __name__ == '__main__':
    # Execute Bronze layer ingestion pipeline
    spark_session = bronze_ingestion()
    
    print("\n[OK] Bronze Layer Ingestion successfully completed!")
    print("Verifying persisted row counts in Bronze Parquet storage:")
    
    acc_out = os.path.join(project_root, "data", "bronze", "accounts_parquet")
    trans_out = os.path.join(project_root, "data", "bronze", "transactions_parquet")
    patt_out = os.path.join(project_root, "data", "bronze", "patterns_parquet")
    
    # Execute count actions to validate storage integrity and trigger physical execution DAG
    acc_count = spark_session.read.parquet(acc_out).count()
    trans_count = spark_session.read.parquet(trans_out).count()
    patt_count = spark_session.read.parquet(patt_out).count()
    
    print(f" - Accounts Bronze Parquet: {acc_count:,} records")
    print(f" - Transactions Bronze Parquet: {trans_count:,} records")
    print(f" - Patterns Bronze Parquet: {patt_count:,} records\n")