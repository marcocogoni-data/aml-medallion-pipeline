"""
main.py - Medallion Architecture Pipeline Master Orchestrator

Description:
    Master orchestration script for the Anti-Money Laundering (AML) PySpark ETL pipeline. 
    Sequentially executes execution steps across Bronze, Silver, and Gold layers within 
    the Medallion Architecture framework, processing 31.8M+ financial records end-to-end.

Architecture & Performance Optimizations:
    - Isolated Subprocess RAM Garbage Collection: Invokes each step via `subprocess.run([sys.executable, ...])` 
      rather than direct Python module imports. This guarantees complete termination of the underlying JVM 
      and driver process at each step boundary, completely purging off-heap memory and preventing OutOfMemoryError (OOM) 
      accumulation over long-running execution pipelines.
    - Deterministic Step Sequencing: Enforces strict dependency execution order (Bronze Raw Ingestion -> 
      Silver Cleansing -> Silver Standardization -> Silver Enrichment -> Silver Graph Analytics -> Gold Aggregation).
    - Pipeline Failure Guardrails: Monitors step return codes (`returncode != 0`) and triggers immediate 
      fail-fast process termination with explicit diagnostics on script failure.
"""

import os
import sys
import subprocess
import time
import logging

# Configure centralized execution logging format for pipeline observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def run_pipeline():
    """
    Executes the end-to-end AML Medallion Data Engineering pipeline sequentially.
    """
    print("=" * 80)
    print("🚀 STARTING END-TO-END AML DATA ENGINEERING PIPELINE (MEDALLION ARCHITECTURE)")
    print("=" * 80)
    
    # Track overall pipeline execution duration
    start_time = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Ordered sequence of execution steps matching the standardized src/ file layout
    pipeline_scripts = [
        ("Stage 1 - Bronze Raw Data Ingestion", "03_bronze_ingestion.py"),
        ("Stage 2.1 - Silver Text Cleansing & Sanitization", "04_cleansing.py"),
        ("Stage 2.2 - Silver Currency Normalization & Timestamp Parsing", "05_currency_timestamp.py"),
        ("Stage 2.3 - Silver Exact Deduplication & Final Export", "06_silver_dedup_export.py"),
        ("Stage 3.1 - Silver Broadcast Join & Entity Enrichment", "07_enrichment_broadcast_join.py"),
        ("Stage 3.2 - Silver Analytical Window Features & Spike Detection", "08_window_features.py"),
        ("Stage 3.3 - Silver Graph Topology Patterns (Fan-In / Fan-Out)", "09_topology_patterns.py"),
        ("Stage 3.4 - Silver Temporal Velocity & Circular Transfer Cycles", "10_temporal_cycle_patterns.py"),
        ("Stage 4.1 - Gold Composite AML Risk Score Aggregation", "11_gold_risk_aggregation.py"),
    ]

    for step_num, (step_name, script_name) in enumerate(pipeline_scripts, start=1):
        script_path = os.path.join(current_dir, script_name)
        logging.info(f"[{step_num}/{len(pipeline_scripts)}] Executing {step_name} ({script_name})...")
        
        # Verify script file existence prior to sub-process spawning
        if not os.path.exists(script_path):
            logging.error(f"❌ Script target not found at absolute path: {script_path}")
            sys.exit(1)
        
        # Execute script in isolated Python sub-process to purge PySpark JVM driver heap after each stage
        result = subprocess.run([sys.executable, script_path])
        
        # Immediate fail-fast strategy on step error
        if result.returncode != 0:
            logging.error(f"❌ Pipeline failure in step '{step_name}' ({script_name}). Return Code: {result.returncode}")
            sys.exit(result.returncode)

    # Compute total pipeline execution time metrics
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 80)
    print(f"🎉 MEDALLION PIPELINE SUCCESSFULLY EXECUTED IN {minutes}m {seconds}s!")
    print("All Bronze, Silver, and Gold Parquet layers successfully updated and certified.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()