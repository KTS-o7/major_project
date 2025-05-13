import argparse
import json
import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()
# Import our timestamp utilities
from log_timestamp_utils import detect_log_format

# Configure LLM client with environment variables
OPENAI_API_KEY = os.getenv("OPENAPI_RC_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAPI_RC_URL")
llm_model = os.getenv("LLM_MODEL")

# Added for rate limiting
DEFAULT_RATE_LIMIT_SECONDS = 2.0  # Default 1 second between requests
MAX_RETRIES = 3  # Maximum number of retries for API calls
BACKOFF_FACTOR = 2  # Exponential backoff factor

# Define data models
class LogChunk(BaseModel):
    """Represents a chunk of log entries."""
    content: List[str] = Field(description="The content of the chunk, each line is a string")
    log_type: str = Field(description="The type of log, ex: HDFS, Android, etc")
    chunk_id: str = Field(description="The id of the chunk")
    num_lines: int = Field(description="The number of lines in the chunk")
    source_file: str = Field(description="The source file of the chunk")


class LogSummary(BaseModel):
    """Represents a summary of a log chunk."""
    summary: str = Field(description="The summary of the chunk")
    errors: Optional[List[str]] = Field(default_factory=list, description="The errors in the chunk")
    warnings: Optional[List[str]] = Field(default_factory=list, description="The warnings in the chunk")
    key_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The key metrics in the chunk")
    severity: str = Field(default="info", description="The severity of the chunk")  # info, warning, error, critical


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

def generate_log_summary(log_chunk: LogChunk, rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS) -> LogSummary:
    """Generate a log summary using the OpenAI API with rate limiting and retries."""
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    
    system_prompt = """You are an expert system administrator and log analyst specializing in infrastructure, application, and system logs. Your task is to analyze and extract structured information from log chunks with precision and technical accuracy.

ANALYSIS GUIDELINES:
1. Extract factual information only - never hallucinate or infer details not explicitly present
2. Provide concise, clear, and technically accurate summaries
3. Identify patterns, sequences, and relationships between log events
4. Recognize standard log patterns across various systems (HDFS, Hadoop, databases, web servers, etc.)
5. Categorize the severity appropriately based on objective criteria

TECHNICAL REQUIREMENTS:
- Parse timestamps, IDs, error codes, and status indicators correctly
- Identify critical system transitions (startup, shutdown, failover events)
- Detect unusual patterns that might indicate issues (repeated errors, timeouts, etc.)
- Recognize sequences that represent workflows or transactions
- Identify cause-effect relationships between log entries when clearly present

RESPONSE FORMAT:
You must output a structured JSON object with the following fields:
1. summary: A technical, factual summary (3-5 sentences) focusing on key activities, state changes, and significant events
2. errors: Complete list of ALL error conditions with their exact error messages and codes
3. warnings: Complete list of ALL warning conditions with their exact warning messages
4. key_metrics: A dictionary containing quantitative data (counts, durations, sizes, IDs) extracted from the logs
5. severity: An assessment of overall severity, strictly one of: ["info", "warning", "error", "critical"]

SEVERITY CRITERIA:
- info: Normal operations, no errors or warnings
- warning: Non-critical issues that don't impact service function but may indicate potential problems
- error: Issues that affect service functionality but don't cause complete failure
- critical: Severe issues causing service downtime, data loss, or security breaches

You must honor the exact format of the log messages when describing errors and warnings.
For exceptional quality, focus on creating summaries that would help a system administrator quickly diagnose issues or understand system behavior."""

    user_prompt = f"""Analyze and summarize the following {log_chunk.log_type} log chunk:

{log_chunk.content}

Source File: {log_chunk.source_file}
Log Type: {log_chunk.log_type}
Number of Lines: {log_chunk.num_lines}
Chunk ID: {log_chunk.chunk_id}

Provide a comprehensive structured analysis with all required fields."""

    # Add retries with exponential backoff
    retry_count = 0
    while retry_count <= MAX_RETRIES:
        try:
            # Add rate limiting with jitter to prevent exact synchronization
            if retry_count > 0:
                sleep_time = rate_limit_seconds * (BACKOFF_FACTOR ** retry_count) + random.uniform(0, 0.5)
                print(f"Retry {retry_count}: Waiting {sleep_time:.2f} seconds before retry...")
                time.sleep(sleep_time)
            else:
                # Add jitter to the rate limit to avoid synchronized requests
                jitter = random.uniform(0, 0.5)
                time.sleep(rate_limit_seconds + jitter)
            
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=60  # 60 second timeout for the API request
            )
            
            response_json = json.loads(response.choices[0].message.content)
            
            # Convert to LogSummary model
            log_summary = LogSummary(
                summary=response_json.get("summary", ""),
                errors=response_json.get("errors", []),
                warnings=response_json.get("warnings", []),
                key_metrics=response_json.get("key_metrics", {}),
                severity=response_json.get("severity", "info")
            )
            
            return log_summary
        
        except Exception as e:
            retry_count += 1
            if retry_count <= MAX_RETRIES:
                print(f"Error during API call: {str(e)}. Retrying ({retry_count}/{MAX_RETRIES})...")
            else:
                print(f"Error generating summary after {MAX_RETRIES} retries: {str(e)}")
                # Return a default summary in case of error
                return LogSummary(
                    summary=f"Failed to generate summary after {MAX_RETRIES} retries: {str(e)}",
                    severity="info"
                )


def extract_log_metadata(
    chunk: List[str], file_path: str, chunk_id: str,
) -> LogChunk:
    """Extract metadata from a log chunk."""
    log_type = determine_log_type(file_path)

    return LogChunk(
        content=chunk,
        log_type=log_type,
        chunk_id=chunk_id,
        num_lines=len(chunk),
        source_file=os.path.basename(file_path),
    )


def determine_log_type(file_path: str) -> str:
    """Determine the type of log based on the file path or content."""
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


def process_log_file(
    file_path: str,
    chunk_method: str = "lines",
    chunk_size: int = 100,
    overlap: int = 0,
    rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
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

    # Chunk the logs
    if chunk_method == "lines":
        chunks = chunk_log_by_lines(lines, chunk_size, overlap)
    else:
        raise ValueError(f"Unknown chunking method: {chunk_method}")

    print(f"Created {len(chunks)} chunks from {file_path}")
    print(f"Rate limiting: {rate_limit_seconds} seconds between API calls (plus jitter)")

    log_entries = []
    error_count = 0

    # Process each chunk with a progress bar
    with tqdm(total=len(chunks), desc="Generating summaries", dynamic_ncols=True) as pbar:
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"{os.path.basename(file_path)}_{i}"

                # Skip chunks that are too short
                if len(chunk) < 5:
                    pbar.update(1)
                    continue

                log_chunk = extract_log_metadata(chunk, file_path, chunk_id)

                # Generate summary using OpenAI
                print("\n--------------------------------\n")
                print(f"Starting API call for chunk {i}...")
                print("LLM model: ", llm_model)
                log_summary = generate_log_summary(log_chunk=log_chunk, rate_limit_seconds=rate_limit_seconds)
                print(f"Log summary: {log_summary.model_dump_json(indent=2, exclude_none=True)}")
                print(f"Completed API call for chunk {i}")
                print("\n--------------------------------\n")
                
                # Create log entry
                log_entry = LogEntry(chunk=log_chunk, summary=log_summary)
                log_entries.append(log_entry)

            except Exception as e:
                error_count += 1
                pbar.set_postfix({"errors": error_count})
                print(f"\nError processing chunk {i}: {str(e)}")
            
            finally:
                pbar.update(1)

    print(f"\nCompleted with {error_count} errors")
    print(f"Generated {len(log_entries)} log chunk-summary pairs")
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
        data = []
        for entry in log_entries:
            entry_dict = entry.model_dump()
            flattened = {
                "chunk_id": entry_dict["chunk"]["chunk_id"],
                "log_type": entry_dict["chunk"]["log_type"],
                "num_lines": entry_dict["chunk"]["num_lines"],
                "source_file": entry_dict["chunk"]["source_file"],
                "log_content": "\n".join(entry_dict["chunk"]["content"]),
                "summary": entry_dict["summary"]["summary"],
                "severity": entry_dict["summary"]["severity"],
                "errors": ", ".join(entry_dict["summary"]["errors"]) if entry_dict["summary"]["errors"] else "",
                "warnings": ", ".join(entry_dict["summary"]["warnings"]) if entry_dict["summary"]["warnings"] else "",
            }
            data.append(flattened)

        df = pd.DataFrame(data)
        output_file = output_path / "log_summaries.csv"
        df.to_csv(output_file, index=False)

    else:
        raise ValueError(f"Unsupported output format: {format}")

    print(f"Dataset saved to {output_file}")

    # Also save a training-ready format for fine-tuning
    training_data = []
    for entry in log_entries:
        entry_dict = entry.model_dump()
        training_data.append({
            "input": "\n".join(entry_dict["chunk"]["content"]),
            "output": entry_dict["summary"]["summary"]
        })

    output_file = output_path / "training_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"Training-ready format saved to {output_file}")


def analyze_dataset(log_entries: List[LogEntry]):
    """Generate statistics about the dataset."""
    if not log_entries:
        print("\nDataset Statistics:")
        print("Total chunks: 0")
        print("Average chunk size: 0.0 lines")
        print("Average summary length: 0.0 words")
        print("Distribution by log type:")
        return

    total_chunks = len(log_entries)
    avg_chunk_size = sum(entry.chunk.num_lines for entry in log_entries) / total_chunks if total_chunks > 0 else 0
    avg_summary_length = sum(len(entry.summary.summary.split()) for entry in log_entries) / total_chunks if total_chunks > 0 else 0

    log_types = {}
    severities = {}
    for entry in log_entries:
        log_type = entry.chunk.log_type
        severity = entry.summary.severity
        
        log_types[log_type] = log_types.get(log_type, 0) + 1
        severities[severity] = severities.get(severity, 0) + 1

    print("\nDataset Statistics:")
    print(f"Total chunks: {total_chunks}")
    print(f"Average chunk size: {avg_chunk_size:.1f} lines")
    print(f"Average summary length: {avg_summary_length:.1f} words")
    
    print("Distribution by log type:")
    for log_type, count in sorted(log_types.items()):
        print(f"  - {log_type}: {count} ({count/total_chunks*100:.1f}%)")
    
    print("Distribution by severity:")
    for severity, count in sorted(severities.items()):
        print(f"  - {severity}: {count} ({count/total_chunks*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate log summarization dataset")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--log_dir", type=str, help="Directory containing log files")
    input_group.add_argument("--log_file", type=str, help="Single log file to process")
    
    # Output options
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    
    # Chunking options
    parser.add_argument(
        "--chunk_method", type=str, choices=["lines", "time"], default="lines",
        help="Method to chunk logs: by line count or time windows"
    )
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of lines per chunk")
    parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping lines between chunks")
    parser.add_argument("--time_window", type=int, default=60, help="Time window in minutes for time-based chunking")
    
    # Processing options
    parser.add_argument("--file_pattern", type=str, default="*.log", help="Pattern to match log files")
    parser.add_argument(
        "--output_format", type=str, choices=["jsonl", "json", "csv"], default="jsonl",
        help="Output format for the dataset"
    )
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--min_chunk_size", type=int, default=5, help="Minimum chunk size to process")
    
    # Rate limiting option
    parser.add_argument("--rate_limit", type=float, default=DEFAULT_RATE_LIMIT_SECONDS, 
                       help="Time in seconds to wait between API calls (default: 1.0)")
    
    args = parser.parse_args()
    
    # Collect log files to process
    log_files = []
    if args.log_dir:
        input_dir = Path(args.log_dir)
        log_files = []
        for pattern in ["**/*.log", "**/*.LOG"]:  # Add more patterns if needed
            log_files.extend(list(input_dir.glob(pattern)))
        if args.max_files:
            log_files = log_files[:args.max_files]
    elif args.log_file:
        log_files = [Path(args.log_file)]
    
    if not log_files:
        print(f"No log files found matching pattern {args.file_pattern}")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Process each log file
    all_log_entries = []
    for file_path in log_files:
        log_entries = process_log_file(
            str(file_path),
            args.chunk_method,
            args.chunk_size,
            args.overlap,
            args.rate_limit,
        )
        all_log_entries.extend(log_entries)
    
    # Save the dataset
    save_dataset(all_log_entries, args.output_dir, args.output_format)
    
    # Analyze the dataset
    analyze_dataset(all_log_entries)


if __name__ == "__main__":
    main()
