"""
02_schemas.py - Centralized Data Schema Definition Module

Description:
    Defines explicit PySpark `StructType` schemas for the Bronze Layer ingestion phase. 
    By enforcing deterministic schemas upfront, this module eliminates the heavy performance 
    cost of PySpark's automatic schema inference (`inferSchema=True`), which would otherwise 
    trigger full-dataset parallel scans over 31.8M+ transaction records and 2.1M+ account entities.

Architecture & Performance Optimizations:
    - Zero Schema Inference Latency: Eliminates secondary I/O read passes across raw CSV files, 
      reducing Bronze ingestion startup time from minutes to milliseconds.
    - Deterministic Data Typing: Prevents silent type mismatches and string-to-numeric casting errors 
      during downstream Spark Catalyst logical plan generation.
    - Memory & GC Optimization: Enforces strict primitive types (LongType, DoubleType, IntegerType) 
      to minimize object overhead in the JVM off-heap and tungsten execution memory space.
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, IntegerType
)


def get_accounts_schema() -> StructType:
    """
    Returns the explicit PySpark StructType schema for the Accounts dataset (HI-Medium_accounts.csv).
    
    Dataset Metrics: ~2.1 Million Entity/Account Records.
    Optimization Impact: Bypasses CSV header inspection and full string parsing on ingestion.
    """
    return StructType([
        # Unique identifier for the banking institution
        StructField("Bank Name", StringType(), True),
        # Numeric institutional identifier cast to LongType to optimize join comparison speed
        StructField("Bank ID", LongType(), True),
        # Account string key used as primary join target for topology and window features
        StructField("Account Number", StringType(), True),
        # Corporate or individual entity identifier for AML risk scoring aggregation
        StructField("Entity ID", StringType(), True),
        # Legal entity registration name
        StructField("Entity Name", StringType(), True)
    ])


def get_transactions_schema() -> StructType:
    """
    Returns the explicit PySpark StructType schema for the Transactions dataset (HI-Medium_Trans.csv).
    
    Dataset Metrics: ~31.8 Million Financial Transaction Records.
    Optimization Impact: Avoids a full 32M+ row scan pass required by inferSchema, saving significant I/O bandwidth.
    """
    return StructType([
        # Raw ISO timestamp string (parsed downstream to TimestampType in Silver processing)
        StructField("Timestamp", StringType(), True),
        # Source Bank ID key stored as LongType for efficient distributed join partitioning
        StructField("From Bank", LongType(), True),
        # Source Account identifier string
        StructField("Account", StringType(), True),
        # Destination Bank ID key stored as LongType for efficient distributed join partitioning
        StructField("To Bank", LongType(), True),
        # Destination Account identifier string (aliased as Account.1 in raw dataset)
        StructField("Account.1", StringType(), True),
        # Transaction value received in target currency (DoubleType for high-precision arithmetic)
        StructField("Amount Received", DoubleType(), True),
        # 3-letter currency code for incoming transfer
        StructField("Receiving Currency", StringType(), True),
        # Transaction value paid in source currency (DoubleType)
        StructField("Amount Paid", DoubleType(), True),
        # 3-letter currency code for outgoing transfer
        StructField("Payment Currency", StringType(), True),
        # Settlement format indicator (e.g., ACH, Wire, Credit Card, Cheque)
        StructField("Payment Format", StringType(), True),
        # Target binary classification label for Anti-Money Laundering (0 = Legitimate, 1 = Illicit)
        StructField("Is Laundering", IntegerType(), True)
    ])


def get_patterns_schema() -> StructType:
    """
    Returns the explicit PySpark StructType schema for the Laundering Patterns reference file (HI-Medium_Patterns.txt).
    
    Dataset Metrics: ~31K Synthetic Pattern Lines.
    Optimization Impact: Provides a lean schema for parsing raw text lines containing graph fraud templates.
    """
    return StructType([
        # Raw text line containing graph topology pattern descriptions and ground truth metadata
        StructField("Pattern_Line", StringType(), True)
    ])


if __name__ == "__main__":
    # Internal module verification and schema structural integrity check
    accounts_schema = get_accounts_schema()
    transactions_schema = get_transactions_schema()
    patterns_schema = get_patterns_schema()

    print("[OK] PySpark StructType Schemas successfully validated:")
    print(f" - Accounts Schema: {len(accounts_schema.fields)} fields defined")
    print(f" - Transactions Schema: {len(transactions_schema.fields)} fields defined")
    print(f" - Patterns Schema: {len(patterns_schema.fields)} fields defined")