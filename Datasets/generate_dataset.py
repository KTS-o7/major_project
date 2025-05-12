#!/usr/bin/env python3
"""
Main script for log summarization dataset generation.
This script combines downloading LogHub datasets and generating summaries in one step.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os
import requests
from tqdm import tqdm
import logging


def run_command(cmd, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n=== {description} ===")

    print(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, text=True)

    # Wait for the process to complete
    return_code = process.wait()

    if return_code != 0:
        print(f"Error: Command failed with exit code {return_code}")
        return False

    return True


def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def download_file(url, destination):
    """
    Download a file from URL to the destination with progress bar.

    Args:
        url: The URL to download from
        destination: The destination file path
    """
    logging.info(f"Starting download from {url} to {destination}")
    session = requests.Session()
    response = session.get(url, allow_redirects=True)
    
    if response.status_code != 200:
        logging.error(f"Failed to access download URL: {response.status_code}")
        raise Exception(f"Failed to access download URL: {response.status_code}")
    
    # Get the final URL after redirects
    final_url = response.url
    
    # Now download the actual file
    response = session.get(final_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    # Check if the file already exists and is complete
    if os.path.exists(destination):
        existing_size = os.path.getsize(destination)
        if existing_size == total_size:
            logging.info(f"File already downloaded and complete: {destination}")
            return
        else:
            logging.warning(f"Incomplete file detected. Removing: {destination}")
            os.remove(destination)

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    desc = f"Downloading {os.path.basename(destination)}"
    with open(destination, "wb") as file, tqdm(
        desc=desc,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        file=sys.stdout,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    logging.info(f"Completed download to {destination}")


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
        default=500,
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
