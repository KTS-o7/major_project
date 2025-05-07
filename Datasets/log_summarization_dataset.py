import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from mirascope.core import openai
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

# Import our timestamp utilities
from .log_timestamp_utils import (
    chunk_by_time_interval,
    detect_log_format,
    extract_time_range,
)

# Configure LLM client
custom_client = OpenAI(api_key="dummy-key", base_url="http://localhost:11434/v1/")
llm_model = "llama3.2:latest"  # You can change this to a different model if needed


# Define data models
class LogChunk(BaseModel):
    """Represents a chunk of log entries."""

    content: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    log_type: str
    chunk_id: str
    num_lines: int
    source_file: str


class LogSummary(BaseModel):
    """Represents a summary of a log chunk."""

    summary: str
    events: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []
    key_metrics: Dict[str, Any] = {}
    severity: str = "info"  # info, warning, error, critical


class LogEntry(BaseModel):
    """Represents a log chunk and its summary for training data."""

    chunk: LogChunk
    summary: LogSummary


def read_log_file(file_path: str) -> List[str]:
    """Read a log file and return its lines."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return lines
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []


def chunk_log_by_lines(
    lines: List[str], chunk_size: int, overlap: int = 0
) -> List[List[str]]:
    """Split log lines into chunks of specified size with optional overlap."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunks.append(lines[i:end])
        i += chunk_size - overlap

    return chunks


def extract_log_metadata(
    chunk: List[str], file_path: str, chunk_id: str, log_format: str = "auto"
) -> LogChunk:
    """Extract metadata from a log chunk, including timestamp information if available."""
    content = "".join(chunk)
    log_type = determine_log_type(file_path)

    # Extract time range if possible
    start_time_dt, end_time_dt = extract_time_range(chunk, log_format)

    start_time = start_time_dt.isoformat() if start_time_dt else None
    end_time = end_time_dt.isoformat() if end_time_dt else None

    return LogChunk(
        content=content,
        start_time=start_time,
        end_time=end_time,
        log_type=log_type,
        chunk_id=chunk_id,
        num_lines=len(chunk),
        source_file=os.path.basename(file_path),
    )


def determine_log_type(file_path: str) -> str:
    """Determine the type of log based on the file path or content."""
    # Basic implementation - customize based on your log sources
    basename = os.path.basename(file_path).lower()

    if "apache" in basename or "nginx" in basename or "httpd" in basename:
        return "web_server"
    elif "mysql" in basename or "postgresql" in basename or "mongo" in basename:
        return "database"
    elif "kernel" in basename or "syslog" in basename:
        return "system"
    elif "auth" in basename or "security" in basename:
        return "security"
    elif "network" in basename or "firewall" in basename:
        return "network"
    elif "hadoop" in basename or "hdfs" in basename:
        return "hadoop"
    elif "zookeeper" in basename:
        return "zookeeper"
    elif "spark" in basename:
        return "spark"
    else:
        return "application"  # Default category


@openai.call(llm_model, response_model=LogSummary, client=custom_client)
def generate_log_summary(
    log_content: str,
    log_type: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> str:
    """
    Generate a summary of a log chunk using the LLM.
    The prompt is engineered to focus on extracting key information from log data.

    Args:
        log_content: The log content to summarize
        log_type: The type of log (web_server, database, system, etc.)
        start_time: The start time of the log chunk (if available)
        end_time: The end time of the log chunk (if available)
    """
    time_info = ""
    if start_time and end_time:
        time_info = f"\nTime period: {start_time} to {end_time}"
    elif start_time:
        time_info = f"\nStart time: {start_time}"
    elif end_time:
        time_info = f"\nEnd time: {end_time}"

    return f"""
You are an expert system administrator analyzing log files. Summarize the following {log_type} log content:{time_info}

{log_content}

Provide a concise summary that includes:
1. Do not make up any information, only use the information provided in the log file.
2. The main events or activities captured in the logs
3. Any errors or warnings that appear
4. Key metrics or performance indicators
5. Overall system status or health

Focus on being factual, precise, and highlight the most important information that a system administrator would need to know.
"""


def process_log_file(
    file_path: str,
    chunk_method: str = "lines",
    chunk_size: int = 100,
    overlap: int = 0,
    time_window: int = 60,
) -> List[LogEntry]:
    """
    Process a log file by:
    1. Reading the file
    2. Chunking it according to the specified method
    3. Generating summaries for each chunk

    Returns a list of LogEntry objects.
    """
    print(f"Processing {file_path}...")
    lines = read_log_file(file_path)

    # Skip empty files
    if not lines:
        print(f"Skipping empty file: {file_path}")
        return []

    # Detect log format for timestamp extraction if needed
    log_format = "auto"
    if chunk_method == "time":
        log_format = detect_log_format(lines)
        if log_format == "unknown":
            print(
                f"Warning: Could not detect timestamp format in {file_path}. Will use line-based chunking."
            )
            chunk_method = "lines"
        else:
            print(f"Detected log format: {log_format}")

    # Chunk the logs
    if chunk_method == "lines":
        chunks = chunk_log_by_lines(lines, chunk_size, overlap)
    elif chunk_method == "time":
        chunks = chunk_by_time_interval(lines, time_window, log_format)
    else:
        raise ValueError(f"Unknown chunking method: {chunk_method}")

    print(f"Created {len(chunks)} chunks from {file_path}")

    log_entries = []

    # Process each chunk
    for i, chunk in enumerate(tqdm(chunks, desc="Generating summaries")):
        chunk_id = f"{os.path.basename(file_path)}_{i}"

        # Skip chunks that are too short
        if len(chunk) < 5:  # Arbitrary minimum size - adjust as needed
            continue

        log_chunk = extract_log_metadata(chunk, file_path, chunk_id, log_format)

        try:
            # Generate summary using LLM
            log_summary = generate_log_summary(
                log_chunk.content,
                log_chunk.log_type,
                log_chunk.start_time,
                log_chunk.end_time,
            )

            # Create log entry
            log_entry = LogEntry(chunk=log_chunk, summary=log_summary)
            log_entries.append(log_entry)

        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")

    return log_entries


def save_dataset(log_entries: List[LogEntry], output_dir: str, format: str = "jsonl"):
    """Save the dataset to disk in the specified format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        output_file = output_path / "log_summaries.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in log_entries:
                f.write(entry.model_dump_json() + "\n")

    elif format == "json":
        output_file = output_path / "log_summaries.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([entry.model_dump() for entry in log_entries], f, indent=2)

    elif format == "csv":
        # Flatten the structure for CSV format
        data = []
        for entry in log_entries:
            entry_dict = entry.model_dump()
            flattened = {
                "chunk_id": entry_dict["chunk"]["chunk_id"],
                "log_type": entry_dict["chunk"]["log_type"],
                "num_lines": entry_dict["chunk"]["num_lines"],
                "source_file": entry_dict["chunk"]["source_file"],
                "start_time": entry_dict["chunk"]["start_time"],
                "end_time": entry_dict["chunk"]["end_time"],
                "log_content": entry_dict["chunk"]["content"],
                "summary": entry_dict["summary"]["summary"],
                "severity": entry_dict["summary"]["severity"],
                "events": ", ".join(entry_dict["summary"]["events"]),
                "errors": ", ".join(entry_dict["summary"]["errors"]),
                "warnings": ", ".join(entry_dict["summary"]["warnings"]),
            }
            data.append(flattened)

        df = pd.DataFrame(data)
        output_file = output_path / "log_summaries.csv"
        df.to_csv(output_file, index=False)

    else:
        raise ValueError(f"Unsupported output format: {format}")

    print(f"Dataset saved to {output_file}")

    # Also save a training_ready format specifically for fine-tuning
    training_data = []
    for entry in log_entries:
        training_data.append(
            {"input": entry.chunk.content, "output": entry.summary.summary}
        )

    output_file = output_path / "training_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"Training-ready format saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate log summarization dataset using LLMs"
    )
    parser.add_argument(
        "--log_dir", type=str, required=True, help="Directory containing log files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for dataset"
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
        "--file_pattern", type=str, default="*.log", help="Pattern to match log files"
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
        help="Maximum number of files to process (for testing)",
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=5,
        help="Minimum number of lines for a valid chunk",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise ValueError(f"Log directory does not exist: {log_dir}")

    # Find all log files matching the pattern
    log_files = list(log_dir.glob(args.file_pattern))
    print(f"Found {len(log_files)} log files")

    # Limit the number of files if specified
    if args.max_files is not None:
        log_files = log_files[: args.max_files]
        print(f"Processing {len(log_files)} files (limited by --max_files)")

    all_log_entries = []

    # Process each log file
    for file_path in log_files:
        log_entries = process_log_file(
            str(file_path),
            chunk_method=args.chunk_method,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            time_window=args.time_window,
        )
        all_log_entries.extend(log_entries)

    print(f"Generated {len(all_log_entries)} log chunk-summary pairs")

    # Save the dataset
    save_dataset(all_log_entries, args.output_dir, format=args.output_format)

    # Print dataset statistics
    print("\nDataset Statistics:")
    log_types = {}
    avg_chunk_size = 0
    avg_summary_length = 0

    for entry in all_log_entries:
        log_type = entry.chunk.log_type
        log_types[log_type] = log_types.get(log_type, 0) + 1
        avg_chunk_size += entry.chunk.num_lines
        avg_summary_length += len(entry.summary.summary.split())

    if all_log_entries:
        avg_chunk_size /= len(all_log_entries)
        avg_summary_length /= len(all_log_entries)

    print(f"Total chunks: {len(all_log_entries)}")
    print(f"Average chunk size: {avg_chunk_size:.1f} lines")
    print(f"Average summary length: {avg_summary_length:.1f} words")
    print("Distribution by log type:")
    for log_type, count in log_types.items():
        print(f"  - {log_type}: {count} chunks ({count/len(all_log_entries)*100:.1f}%)")


if __name__ == "__main__":
    main()
