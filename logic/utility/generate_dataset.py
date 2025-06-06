import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional

import coloredlogs
import instructor
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field

load_dotenv()

# Set up logging with more detailed format
logger = logging.getLogger("log_processor")
coloredlogs.install(
    level="INFO",
    logger=logger,
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level_styles={
        "debug": {"color": "white"},
        "info": {"color": "green"},
        "warning": {"color": "yellow"},
        "error": {"color": "red", "bold": True},
        "critical": {"color": "red", "bold": True, "background": "white"},
    },
)

# Add color definitions after coloredlogs.install
RESET = "\033[0m"
REQ_COLOR = "\033[94m"  # Bright Blue for request logs
RESP_COLOR = "\033[92m"  # Bright Green for response logs
INFO_COLOR = "\033[96m"  # Bright Cyan for general info logs
ERROR_COLOR = "\033[91m"  # Bright Red for error logs

# Initialize instructor with litellm
client = instructor.from_litellm(completion)

# Rate limiting configuration
REQUESTS_PER_MINUTE = 15
REQUESTS_PER_DAY = 1500
TOKENS_PER_MINUTE = 1000000

# Data directory
DATA_DIR = "data"

# Output directory
OUTPUT_DIR = "dataset"

# Use multiple API keys in sequence to prevent rate limiting
API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]
API_KEYS = [key for key in API_KEYS if key is not None]  # Filter out None values


class RateLimiter:
    def __init__(
        self, requests_per_minute: int, requests_per_day: int, tokens_per_minute: int
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.tokens_per_minute = tokens_per_minute

        # Track rate limits per API key
        self.key_limits = {
            key: {
                "minute_requests": [],
                "day_requests": [],
                "minute_tokens": [],
                "is_exhausted": False,
                "last_request_time": None,  # Track last request time
            }
            for key in API_KEYS
        }

        self.current_key_index = 0
        logger.info(f"Initialized RateLimiter with {len(API_KEYS)} API keys")

    def get_current_api_key(self) -> str:
        """Get current API key that hasn't exceeded rate limits."""
        for _ in range(len(API_KEYS)):
            key = API_KEYS[self.current_key_index % len(API_KEYS)]
            if not self.key_limits[key]["is_exhausted"]:
                return key
            self.current_key_index += 1

        # If all keys are exhausted, reset the first key and return it
        logger.warning("All API keys exhausted, resetting first key")
        self.current_key_index = 0
        self.key_limits[API_KEYS[0]]["is_exhausted"] = False
        return API_KEYS[0]

    def rotate_api_key(self):
        """Rotate to next available API key."""
        self.current_key_index += 1
        next_key = API_KEYS[self.current_key_index % len(API_KEYS)]
        logger.info(
            f"Rotating to API key {(self.current_key_index % len(API_KEYS)) + 1}"
        )

        # Reset the key's limits if it was exhausted
        if self.key_limits[next_key]["is_exhausted"]:
            logger.debug(f"Resetting limits for key {next_key[:8]}...")
            self.key_limits[next_key]["is_exhausted"] = False
            self.key_limits[next_key]["minute_requests"] = []
            self.key_limits[next_key]["day_requests"] = []
            self.key_limits[next_key]["minute_tokens"] = []

    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if rate limits would be exceeded for the current key."""
        current_key = self.get_current_api_key()
        now = datetime.now()

        # Clean old requests for current key
        self.key_limits[current_key]["minute_requests"] = [
            req
            for req in self.key_limits[current_key]["minute_requests"]
            if now - req < timedelta(minutes=1)
        ]
        self.key_limits[current_key]["day_requests"] = [
            req
            for req in self.key_limits[current_key]["day_requests"]
            if now - req < timedelta(days=1)
        ]
        self.key_limits[current_key]["minute_tokens"] = [
            (req, tokens)
            for req, tokens in self.key_limits[current_key]["minute_tokens"]
            if now - req < timedelta(minutes=1)
        ]

        # Calculate time since last request
        last_request_time = self.key_limits[current_key]["last_request_time"]
        if last_request_time:
            time_since_last_request = (now - last_request_time).total_seconds()
            # Wait 4 seconds between requests (15 requests per minute)
            if time_since_last_request < 4:
                wait_time = 4 - time_since_last_request
                logger.info(
                    f"Waiting {wait_time:.2f} seconds to maintain rate limit..."
                )
                logger.info(
                    json.dumps(
                        {
                            "wait_time": wait_time,
                            "time_since_last_request": time_since_last_request,
                            "current_key": current_key[:8] + "...",
                            "timestamp": now.isoformat(),
                        },
                        indent=2,
                    )
                )
                time.sleep(wait_time)

        # Check minute limits
        if (
            len(self.key_limits[current_key]["minute_requests"])
            >= self.requests_per_minute
        ):
            wait_time = (
                60
                - (now - min(self.key_limits[current_key]["minute_requests"])).seconds
            )
            logger.warning(
                f"Rate limit reached for key {current_key[:8]}... Waiting {wait_time} seconds..."
            )
            time.sleep(wait_time + 1)

        # Check daily limits
        if len(self.key_limits[current_key]["day_requests"]) >= self.requests_per_day:
            logger.error(f"Daily rate limit reached for key {current_key[:8]}...")
            self.key_limits[current_key]["is_exhausted"] = True
            self.rotate_api_key()

        # Check token limits
        current_minute_tokens = sum(
            tokens for _, tokens in self.key_limits[current_key]["minute_tokens"]
        )
        if current_minute_tokens + estimated_tokens > self.tokens_per_minute:
            wait_time = (
                60
                - (
                    now
                    - min(
                        req for req, _ in self.key_limits[current_key]["minute_tokens"]
                    )
                ).seconds
            )
            logger.warning(
                f"Token rate limit reached for key {current_key[:8]}... Waiting {wait_time} seconds..."
            )
            time.sleep(wait_time + 1)

    def record_request(self, tokens_used: int = 1000):
        """Record a request for the current API key."""
        current_key = self.get_current_api_key()
        now = datetime.now()

        self.key_limits[current_key]["minute_requests"].append(now)
        self.key_limits[current_key]["day_requests"].append(now)
        self.key_limits[current_key]["minute_tokens"].append((now, tokens_used))
        self.key_limits[current_key][
            "last_request_time"
        ] = now  # Update last request time

        # Log request recording with color highlight
        logger.info(f"{INFO_COLOR}Recorded API request:{RESET}")
        logger.info(
            json.dumps(
                {
                    "current_key": current_key[:8] + "...",
                    "tokens_used": tokens_used,
                    "requests_this_minute": len(
                        self.key_limits[current_key]["minute_requests"]
                    ),
                    "requests_today": len(self.key_limits[current_key]["day_requests"]),
                    "timestamp": now.isoformat(),
                },
                indent=2,
            )
        )

        # Check if we need to rotate keys
        if (
            len(self.key_limits[current_key]["day_requests"]) >= self.requests_per_day
            or len(self.key_limits[current_key]["minute_requests"])
            >= self.requests_per_minute
        ):
            self.key_limits[current_key]["is_exhausted"] = True
            self.rotate_api_key()


# Define the structured output schema using Pydantic (better than dataclass)
class LogSummary(BaseModel):
    system: str = Field(
        description="Component/Service Name with host/instance if relevant"
    )
    operation: str = Field(description="What operation was being performed")
    status: str = Field(
        description="Status with emoji: ✅ Success | ⚠️ Warning | ❌ Failed | 🔄 In Progress"
    )
    duration: str = Field(description="Time span covered by the logs")
    summary: str = Field(description="2-3 sentence overview of main events")
    issues: str = Field(description="Any problems encountered, or 'None detected'")
    impact: str = Field(description="Business/operational impact")


def get_log_files_sequential() -> Iterator[str]:
    """Get log files sequentially from data directory, folder by folder."""
    data_dir = Path(DATA_DIR)
    log_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not log_dirs:
        logger.error("No log directories found in data/")
        raise ValueError("No log directories found in data/")

    for log_dir in log_dirs:
        log_files = sorted(list(log_dir.glob("*.log")))
        if log_files:
            logger.info(f"Processing directory: {log_dir.name}")
            for log_file in log_files:
                yield str(log_file)


def read_log_lines(log_file: str, chunk_size: int = 10) -> Iterator[List[str]]:
    """Read log file in chunks of specified size."""
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for line in f:
                lines.append(line)
                if len(lines) >= chunk_size:
                    yield lines
                    lines = []
            # Yield any remaining lines
            if lines:
                yield lines
            logger.debug(f"Finished reading {log_file}")
    except Exception as e:
        logger.error(f"Error reading {log_file}: {e}")
        return []


def estimate_tokens(text: str) -> int:
    """Estimate token count for the text."""
    # Simple estimation: ~4 characters per token
    return len(text) // 4


def generate_summary(
    log_lines: List[str], rate_limiter: RateLimiter
) -> Optional[LogSummary]:
    """Generate a structured summary using Gemini 2.0 Flash with rate limiting."""

    log_text = "".join(log_lines)
    estimated_tokens = len(log_text) // 4 + 500  # Add buffer for response

    # Log request details with color highlight
    logger.info(f"{REQ_COLOR}Preparing API request:{RESET}")
    logger.info(
        json.dumps(
            {
                "estimated_tokens": estimated_tokens,
                "log_lines_count": len(log_lines),
                "current_api_key": rate_limiter.get_current_api_key()[:8] + "...",
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        )
    )

    # Wait if rate limits would be exceeded
    rate_limiter.wait_if_needed(estimated_tokens)

    try:
        os.environ["GOOGLE_API_KEY"] = rate_limiter.get_current_api_key()

        # Log the actual request with color highlight
        logger.info(f"{REQ_COLOR}Sending request to Gemini API:{RESET}")
        logger.info(
            json.dumps(
                {
                    "model": "gemini/gemini-2.0-flash",
                    "temperature": 0.1,
                    "max_retries": 3,
                    "request_timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
        )

        summary = client.chat.completions.create(
            model="gemini/gemini-2.0-flash",
            response_model=LogSummary,
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze these log lines and generate a structured summary:\n{log_text}",
                }
            ],
            max_retries=3,
            temperature=0.1,
        )

        # Log successful response with color highlight
        logger.info(f"{RESP_COLOR}Received successful response:{RESET}")
        logger.info(
            json.dumps(
                {
                    "system": summary.system,
                    "operation": summary.operation,
                    "status": summary.status,
                    "duration": summary.duration,
                    "response_timestamp": datetime.now().isoformat(),
                },
                indent=2,
            )
        )

        rate_limiter.record_request(estimated_tokens)
        return summary

    except Exception as e:
        logger.error(f"{ERROR_COLOR}API request failed:{RESET}")
        logger.error(
            json.dumps(
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat(),
                    "current_api_key": rate_limiter.get_current_api_key()[:8] + "...",
                },
                indent=2,
            )
        )
        rate_limiter.rotate_api_key()
        return None


def create_dataset_entry(log_lines: List[str], summary: LogSummary) -> dict:
    """Create a dataset entry with instruction, input, and output."""
    return {
        "instruction": "Analyze the following log lines and generate a structured summary.",
        "input": "".join(log_lines),
        "output": f"""**System**: {summary.system}
**Operation**: {summary.operation}
**Status**: {summary.status}
**Duration**: {summary.duration}
**Summary**: {summary.summary}
**Issues**: {summary.issues}
**Impact**: {summary.impact}""",
    }


def save_entry_to_jsonl(entry: dict, output_file: str):
    """Append a single entry to a JSONL file."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_existing_entries(jsonl_file: str) -> set:
    """Load existing entries to avoid duplicates."""
    processed_files = set()
    if os.path.exists(jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Assuming the input field contains the log content
                    processed_files.add(entry["input"])
                except json.JSONDecodeError:
                    continue
    return processed_files


def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize rate limiter
    rate_limiter = RateLimiter(REQUESTS_PER_MINUTE, REQUESTS_PER_DAY, TOKENS_PER_MINUTE)

    # Set up JSONL file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/log_summary_dataset_{timestamp}.jsonl"

    logger.info("Starting dataset generation:")
    logger.info(
        json.dumps(
            {
                "output_file": output_file,
                "rate_limits": {
                    "requests_per_minute": REQUESTS_PER_MINUTE,
                    "requests_per_day": REQUESTS_PER_DAY,
                    "tokens_per_minute": TOKENS_PER_MINUTE,
                },
                "start_time": datetime.now().isoformat(),
            },
            indent=2,
        )
    )

    # Load already processed entries
    processed_entries = load_existing_entries(output_file)

    # Process log files sequentially
    for log_file in get_log_files_sequential():
        try:
            logger.info(f"Processing file: {log_file}")

            # Process the entire log file in chunks
            for chunk_index, log_lines in enumerate(
                read_log_lines(log_file, chunk_size=10)
            ):
                # Log chunk processing
                logger.info(f"Processing chunk {chunk_index + 1} from {log_file}")
                logger.info(
                    json.dumps(
                        {
                            "chunk_index": chunk_index + 1,
                            "lines_in_chunk": len(log_lines),
                            "timestamp": datetime.now().isoformat(),
                        },
                        indent=2,
                    )
                )

                # Check if we've already processed this log content
                log_content = "".join(log_lines)
                if log_content in processed_entries:
                    logger.debug(f"Skipping already processed chunk from: {log_file}")
                    continue

                # Generate summary
                summary = generate_summary(log_lines, rate_limiter)
                if not summary:
                    continue

                # Create dataset entry
                entry = create_dataset_entry(log_lines, summary)

                # Check token count
                total_tokens = estimate_tokens(str(entry))
                if total_tokens < 8000:
                    # Save entry immediately to JSONL
                    save_entry_to_jsonl(entry, output_file)
                    processed_entries.add(log_content)
                    logger.info("Added new entry to dataset:")
                    logger.info(
                        json.dumps(
                            {
                                "tokens": total_tokens,
                                "entry_count": len(processed_entries),
                                "timestamp": datetime.now().isoformat(),
                            },
                            indent=2,
                        )
                    )
                else:
                    logger.warning(f"Skipped entry - too many tokens: {total_tokens}")

        except Exception as e:
            logger.error(f"Error processing {log_file}:")
            logger.error(
                json.dumps(
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "file": log_file,
                        "timestamp": datetime.now().isoformat(),
                    },
                    indent=2,
                )
            )
            continue

    logger.info("Dataset generation completed:")
    logger.info(
        json.dumps(
            {
                "output_file": output_file,
                "total_entries": len(processed_entries),
                "end_time": datetime.now().isoformat(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
