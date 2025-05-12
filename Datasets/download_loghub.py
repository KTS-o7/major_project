import argparse
import glob
import os
import shutil
import sys
import tarfile
import zipfile

import requests
from tqdm import tqdm

# LogHub dataset resources - URLs for datasets from logpai/loghub repository
LOGHUB_DATASETS = {
    "HDFS": "https://zenodo.org/records/8196385/files/HDFS.tar.gz?download=1",
    "Hadoop": "https://zenodo.org/records/8196385/files/Hadoop.tar.gz?download=1",
    "Spark": "https://zenodo.org/records/8196385/files/Spark.tar.gz?download=1",
    "Zookeeper": "https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1",
    "BGL": "https://zenodo.org/records/8196385/files/BGL.tar.gz?download=1",
    "HPC": "https://zenodo.org/records/8196385/files/HPC.tar.gz?download=1",
    "Thunderbird": "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1",
    "Windows": "https://zenodo.org/records/8196385/files/Windows.tar.gz?download=1",
    "Linux": "https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1",
    "Android": "https://zenodo.org/records/8196385/files/Android.tar.gz?download=1",
    "HealthApp": "https://zenodo.org/records/8196385/files/HealthApp.tar.gz?download=1",
    "Apache": "https://zenodo.org/records/8196385/files/Apache.tar.gz?download=1",
    "OpenSSH": "https://zenodo.org/records/8196385/files/SSH.tar.gz?download=1",
    "OpenStack": "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1",
    "Mac": "https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1",
    "Proxifier": "https://zenodo.org/records/8196385/files/Proxifier.tar.gz?download=1",
}


def download_file(url, destination):
    """
    Download a file from URL to the destination with progress bar.

    Args:
        url: The URL to download from
        destination: The destination file path
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

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


def extract_archive(archive_path, extract_dir):
    """
    Extract a zip or tar.gz archive to the specified directory.

    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to
    """
    os.makedirs(extract_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for file in tqdm(
                zip_ref.namelist(), desc=f"Extracting {os.path.basename(archive_path)}"
            ):
                zip_ref.extract(file, extract_dir)
    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            for file in tqdm(
                tar_ref.getmembers(),
                desc=f"Extracting {os.path.basename(archive_path)}",
            ):
                tar_ref.extract(file, extract_dir)
    else:
        print(f"Unsupported archive format: {archive_path}")


def organize_extracted_files(extract_dir, dataset_name):
    """
    Organize extracted files into a consistent structure.
    Different LogHub datasets have different structures after extraction.

    Args:
        extract_dir: Directory where files were extracted
        dataset_name: Name of the dataset (e.g., "HDFS")

    Returns:
        Path to the main log file
    """
    # Find all log files
    log_files = []
    for ext in [".log", ".LOG"]:
        log_files.extend(glob.glob(f"{extract_dir}/**/*{ext}", recursive=True))

    if not log_files:
        print(f"No log files found in {extract_dir}")
        return None

    # Create a logs directory for the dataset
    logs_dir = os.path.join(extract_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Move or copy log files to the logs directory
    for log_file in log_files:
        dest_file = os.path.join(logs_dir, os.path.basename(log_file))
        if not os.path.exists(dest_file):
            shutil.copy2(log_file, dest_file)
            print(f"Copied {os.path.basename(log_file)} to {logs_dir}")

    # For HDFS dataset, there might be special handling needed
    if dataset_name == "HDFS":
        hdfs_logs = glob.glob(f"{extract_dir}/**/HDFS*.log", recursive=True)
        if hdfs_logs:
            return hdfs_logs[0]

    # Return the path to the main log file
    main_log_files = glob.glob(f"{logs_dir}/*_{dataset_name}*.log") or glob.glob(
        f"{logs_dir}/*{dataset_name}*.log"
    )
    if main_log_files:
        return main_log_files[0]
    elif log_files:
        # Just return the first log file if we couldn't find a specific one
        return log_files[0]
    else:
        return None


def download_dataset(dataset_name, output_dir, extract=True):
    """
    Download a LogHub dataset and optionally extract it.

    Args:
        dataset_name: Name of the dataset (e.g., "HDFS")
        output_dir: Directory to save the dataset
        extract: Whether to extract the archive after downloading

    Returns:
        Path to the downloaded/extracted dataset
    """
    if dataset_name not in LOGHUB_DATASETS:
        available = ", ".join(LOGHUB_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        )

    url = LOGHUB_DATASETS[dataset_name]
    filename = url.split("/")[-1]

    # Create dataset-specific directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download path
    download_path = os.path.join(dataset_dir, filename)

    # Download if the file doesn't exist
    if not os.path.exists(download_path):
        print(f"Downloading {dataset_name} dataset...")
        download_file(url, download_path)
    else:
        print(f"{dataset_name} dataset already downloaded to {download_path}")

    if extract:
        extract_dir = os.path.join(dataset_dir, "raw")

        # Extract if the directory doesn't exist or is empty
        if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
            print(f"Extracting {dataset_name} dataset...")
            extract_archive(download_path, extract_dir)
            main_log_file = organize_extracted_files(extract_dir, dataset_name)
            if main_log_file:
                print(f"Main log file: {main_log_file}")
        else:
            print(f"{dataset_name} dataset already extracted to {extract_dir}")

        return extract_dir

    return download_path


def main():
    parser = argparse.ArgumentParser(description="Download datasets from LogHub")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["HDFS"],
        help="Names of datasets to download (e.g., HDFS, Hadoop, Spark)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data", help="Directory to save datasets"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract archives after downloading",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available LogHub datasets:")
        for name in sorted(LOGHUB_DATASETS.keys()):
            print(f"  - {name}")
        return

    for dataset_name in args.datasets:
        try:
            path = download_dataset(
                dataset_name, args.output_dir, extract=not args.no_extract
            )
            print(f"Successfully processed {dataset_name} dataset at {path}")
        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {str(e)}")


if __name__ == "__main__":
    main()
