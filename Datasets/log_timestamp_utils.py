import re
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Tuple

# Timestamp extraction patterns for common log formats
TIMESTAMP_PATTERNS = {
    # Standard ISO format: 2023-04-24T12:34:56
    "iso": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2})?)",
    # Common syslog format: Apr 24 12:34:56
    "syslog": r"([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})",
    # Apache/Nginx format: 24/Apr/2023:12:34:56 +0000
    "apache": r"(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})",
    # Simple date-time: 2023-04-24 12:34:56
    "simple": r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",
    # Windows event log: 4/24/2023 12:34:56 PM
    "windows": r"(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M)",
    # Java/Tomcat log: 24-Apr-2023 12:34:56.789
    "java": r"(\d{2}-[A-Z][a-z]{2}-\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{3})",
}

# Date format strings for parsing the extracted timestamps
FORMAT_STRINGS = {
    "iso": "%Y-%m-%dT%H:%M:%S",  # May need adjustment for milliseconds and timezone
    "syslog": "%b %d %H:%M:%S",  # Doesn't include year, will need special handling
    "apache": "%d/%b/%Y:%H:%M:%S %z",
    "simple": "%Y-%m-%d %H:%M:%S",
    "windows": "%m/%d/%Y %I:%M:%S %p",
    "java": "%d-%b-%Y %H:%M:%S.%f",  # May need to truncate milliseconds
}


def create_timestamp_extractor(
    log_format: str = "auto",
) -> Callable[[str], Optional[datetime]]:
    """
    Create a function that extracts timestamps from log lines based on the specified format.

    Args:
        log_format: The format to use ("iso", "syslog", "apache", "simple", "windows", "java"),
                   or "auto" to try all formats.

    Returns:
        A function that takes a log line and returns a datetime object or None if no timestamp found.
    """

    def extractor(line: str) -> Optional[datetime]:
        # For auto detection, try all patterns
        if log_format == "auto":
            for fmt, pattern in TIMESTAMP_PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    try:
                        timestamp_str = match.group(1)
                        return parse_timestamp(timestamp_str, fmt)
                    except ValueError:
                        continue
            return None

        # For specific format
        elif log_format in TIMESTAMP_PATTERNS:
            pattern = TIMESTAMP_PATTERNS[log_format]
            match = re.search(pattern, line)
            if match:
                try:
                    timestamp_str = match.group(1)
                    return parse_timestamp(timestamp_str, log_format)
                except ValueError:
                    return None
            return None

        else:
            raise ValueError(f"Unknown log format: {log_format}")

    return extractor


def parse_timestamp(timestamp_str: str, format_name: str) -> datetime:
    """
    Parse a timestamp string based on the format name.

    Args:
        timestamp_str: The timestamp string to parse
        format_name: The name of the format to use

    Returns:
        A datetime object
    """
    format_string = FORMAT_STRINGS[format_name]

    # Special handling for syslog format which doesn't include the year
    if format_name == "syslog":
        current_year = datetime.now().year
        dt = datetime.strptime(timestamp_str, format_string)
        # Add the current year
        dt = dt.replace(year=current_year)

        # Handle year rollover (if the parsed date is in the future by more than a day)
        if dt > datetime.now() + timedelta(days=1):
            dt = dt.replace(year=current_year - 1)

        return dt

    # Handle milliseconds in ISO format
    elif format_name == "iso" and "." in timestamp_str:
        # Split at timezone indicator if present
        if "+" in timestamp_str or "-" in timestamp_str:
            if "+" in timestamp_str:
                main_part = timestamp_str.split("+")[0]
            else:
                # Find the last '-' which is likely the timezone separator
                main_part = timestamp_str[: timestamp_str.rindex("-")]

            if "." in main_part:
                dt_str, ms_str = main_part.split(".")
                ms_str = ms_str[:6]  # Truncate to microseconds
                ms_str = ms_str.ljust(6, "0")  # Pad to 6 digits
                format_string = "%Y-%m-%dT%H:%M:%S.%f"
                dt = datetime.strptime(f"{dt_str}.{ms_str}", format_string)

                # Add timezone info if needed (simplified)
                # This is a basic implementation - for production use consider using pytz or dateutil
                return dt
            else:
                return datetime.strptime(main_part, "%Y-%m-%dT%H:%M:%S")
        else:
            if "." in timestamp_str:
                format_string = "%Y-%m-%dT%H:%M:%S.%f"
            return datetime.strptime(timestamp_str, format_string)

    # Standard parsing for other formats
    return datetime.strptime(timestamp_str, format_string)


def time_diff_minutes(dt1: datetime, dt2: datetime) -> float:
    """
    Calculate the time difference between two datetime objects in minutes.

    Args:
        dt1: First datetime
        dt2: Second datetime

    Returns:
        Time difference in minutes (absolute value)
    """
    diff = abs(dt2 - dt1)
    return diff.total_seconds() / 60.0


def detect_log_format(log_lines: List[str], sample_size: int = 100) -> str:
    """
    Detect the most likely log format from a list of log lines.

    Args:
        log_lines: List of log lines
        sample_size: Number of lines to sample for detection

    Returns:
        The detected format name or "unknown" if no format matches
    """
    sample = log_lines[: min(sample_size, len(log_lines))]
    format_matches = {fmt: 0 for fmt in TIMESTAMP_PATTERNS.keys()}

    for line in sample:
        for fmt, pattern in TIMESTAMP_PATTERNS.items():
            if re.search(pattern, line):
                format_matches[fmt] += 1

    # Get the format with the most matches
    best_format = max(format_matches.items(), key=lambda x: x[1])

    # If at least 10% of the lines match a format, return it
    if best_format[1] >= len(sample) * 0.1:
        return best_format[0]
    else:
        return "unknown"


def chunk_by_time_interval(
    log_lines: List[str], time_window_minutes: int, log_format: str = "auto"
) -> List[List[str]]:
    """
    Chunk log lines by time interval.

    Args:
        log_lines: List of log lines
        time_window_minutes: Size of the time window in minutes
        log_format: The log format to use for timestamp extraction

    Returns:
        List of chunks, where each chunk is a list of log lines
    """
    # Auto-detect format if not specified
    if log_format == "auto":
        detected_format = detect_log_format(log_lines)
        if detected_format == "unknown":
            # Fall back to line-based chunking if format can't be detected
            print(
                "Warning: Could not detect timestamp format. Falling back to line-based chunking."
            )
            chunk_size = 100  # Default chunk size
            return [
                log_lines[i : i + chunk_size]
                for i in range(0, len(log_lines), chunk_size)
            ]
        else:
            print(f"Detected log format: {detected_format}")
            log_format = detected_format

    # Create timestamp extractor for the format
    extract_timestamp = create_timestamp_extractor(log_format)

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_timestamp = None

    for line in log_lines:
        timestamp = extract_timestamp(line)

        # If no timestamp, add to current chunk if exists
        if timestamp is None:
            if current_chunk:
                current_chunk.append(line)
            continue

        # Start new chunk if this is the first line with timestamp
        if current_timestamp is None:
            current_timestamp = timestamp
            current_chunk = [line]
            continue

        # Check if time window is exceeded
        time_diff = time_diff_minutes(current_timestamp, timestamp)

        if time_diff > time_window_minutes:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [line]
            current_timestamp = timestamp
        else:
            current_chunk.append(line)

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def extract_time_range(
    chunk: List[str], log_format: str = "auto"
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Extract the start and end time from a chunk of log lines.

    Args:
        chunk: List of log lines
        log_format: The log format to use

    Returns:
        Tuple of (start_time, end_time) as datetime objects, or None if not found
    """
    extract_timestamp = create_timestamp_extractor(log_format)

    timestamps = []
    for line in chunk:
        ts = extract_timestamp(line)
        if ts is not None:
            timestamps.append(ts)

    if not timestamps:
        return None, None

    return min(timestamps), max(timestamps)
