#!/usr/bin/env python3
"""
Main script for log summarization dataset generation.
This script combines downloading LogHub datasets and generating summaries in one step.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n=== {description} ===")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False

    print(result.stdout)
    return True


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def generate_dataset(args):
    """Generate a log summarization dataset from LogHub."""
    # Create base directories
    data_dir = Path(args.output_dir)
    ensure_directory(data_dir)

    # Step 1: Download and extract datasets
    datasets_arg = []
    for dataset in args.datasets:
        datasets_arg.extend(["--datasets", dataset])

    cmd = [
        sys.executable,
        "download_loghub.py",
        *datasets_arg,
        "--output_dir",
        str(data_dir),
    ]

    if not run_command(cmd, "Downloading LogHub datasets"):
        return False

    # Step 2: Generate summaries for each dataset
    for dataset in args.datasets:
        dataset_raw_dir = data_dir / dataset / "raw" / "logs"
        dataset_processed_dir = data_dir / dataset / "processed"

        # Skip if the dataset directory doesn't exist
        if not dataset_raw_dir.exists():
            print(f"Warning: Raw logs directory not found for {dataset}. Skipping...")
            continue

        ensure_directory(dataset_processed_dir)

        cmd = [
            sys.executable,
            "log_summarization_dataset.py",
            "--log_dir",
            str(dataset_raw_dir),
            "--output_dir",
            str(dataset_processed_dir),
            "--chunk_method",
            args.chunk_method,
            "--output_format",
            args.output_format,
        ]

        if args.chunk_method == "lines":
            cmd.extend(["--chunk_size", str(args.chunk_size)])
            cmd.extend(["--overlap", str(args.overlap)])
        else:  # time-based chunking
            cmd.extend(["--time_window", str(args.time_window)])

        if args.max_files:
            cmd.extend(["--max_files", str(args.max_files)])

        desc = f"Generating summaries for {dataset} dataset"
        if not run_command(cmd, desc):
            print(f"Warning: Failed to process {dataset} dataset. Continuing...")

    print("\n=== Dataset Generation Complete ===")
    print(f"Datasets have been generated in: {data_dir}")
    print("Each dataset contains:")
    print("  - log_summaries.jsonl/json/csv: Full dataset with chunks and summaries")
    print("  - training_data.json: Ready-to-use for fine-tuning (input/output pairs)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate log summarization datasets from LogHub"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["HDFS"],
        help="Names of datasets to download (e.g., HDFS, Hadoop, Spark)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Base directory to save datasets",
    )
    parser.add_argument(
        "--chunk_method",
        type=str,
        default="lines",
        choices=["lines", "time"],
        help="Method to chunk logs (by line count or time windows)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Number of lines per chunk (for line-based chunking)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of overlapping lines between chunks",
    )
    parser.add_argument(
        "--time_window",
        type=int,
        default=60,
        help="Time window in minutes (for time-based chunking)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "csv"],
        help="Output format for the dataset",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process per dataset (for testing)",
    )

    args = parser.parse_args()

    if not generate_dataset(args):
        print("Dataset generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
