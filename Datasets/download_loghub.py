import argparse
import os
import sys

import requests
from tqdm import tqdm

# LogHub 2k dataset resources - URLs for smaller log files from logpai/loghub GitHub repository
LOGHUB_2K_DATASETS = {
    "HDFS": "https://github.com/logpai/loghub/raw/master/HDFS/HDFS_2k.log",
    "Hadoop": "https://github.com/logpai/loghub/raw/master/Hadoop/Hadoop_2k.log",
    "Spark": "https://github.com/logpai/loghub/raw/master/Spark/Spark_2k.log",
    "Zookeeper": "https://github.com/logpai/loghub/raw/master/Zookeeper/Zookeeper_2k.log",
    "BGL": "https://github.com/logpai/loghub/raw/master/BGL/BGL_2k.log",
    "HPC": "https://github.com/logpai/loghub/raw/master/HPC/HPC_2k.log",
    "Thunderbird": "https://github.com/logpai/loghub/raw/master/Thunderbird/Thunderbird_2k.log",
    "Windows": "https://github.com/logpai/loghub/raw/master/Windows/Windows_2k.log",
    "Linux": "https://github.com/logpai/loghub/raw/master/Linux/Linux_2k.log",
    "Android": "https://github.com/logpai/loghub/raw/master/Android/Android_2k.log",
    "HealthApp": "https://github.com/logpai/loghub/raw/master/HealthApp/HealthApp_2k.log",
    "Apache": "https://github.com/logpai/loghub/raw/master/Apache/Apache_2k.log",
    "OpenSSH": "https://github.com/logpai/loghub/raw/master/OpenSSH/OpenSSH_2k.log",
    "OpenStack": "https://github.com/logpai/loghub/raw/master/OpenStack/OpenStack_2k.log",
    "Mac": "https://github.com/logpai/loghub/raw/master/Mac/Mac_2k.log",
    "Proxifier": "https://github.com/logpai/loghub/raw/master/Proxifier/Proxifier_2k.log",
}


def download_file(url, destination):
    """
    Download a file from URL to the destination with progress bar.

    Args:
        url: The URL to download from
        destination: The destination file path
    """
    response = requests.get(url, stream=True)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}, status code: {response.status_code}")
    
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


def download_2k_dataset(dataset_name, output_dir):
    """
    Download a 2k line LogHub dataset directly from GitHub.

    Args:
        dataset_name: Name of the dataset (e.g., "HDFS")
        output_dir: Directory to save the dataset

    Returns:
        Path to the downloaded log file
    """
    if dataset_name not in LOGHUB_2K_DATASETS:
        available = ", ".join(LOGHUB_2K_DATASETS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        )

    url = LOGHUB_2K_DATASETS[dataset_name]
    filename = f"{dataset_name}_2k.log"

    # Create dataset-specific directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download path
    download_path = os.path.join(dataset_dir, filename)

    # Download if the file doesn't exist
    if not os.path.exists(download_path):
        print(f"Downloading {dataset_name} 2k dataset...")
        download_file(url, download_path)
    else:
        print(f"{dataset_name} 2k dataset already downloaded to {download_path}")

    return download_path


def main():
    parser = argparse.ArgumentParser(description="Download 2k datasets from LogHub GitHub")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["HDFS"],
        help="Names of 2k datasets to download (e.g., HDFS, Hadoop, Spark)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="Directory to save datasets"
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        print("Available LogHub 2k datasets:")
        for name in sorted(LOGHUB_2K_DATASETS.keys()):
            print(f"  - {name}")
        return

    for dataset_name in args.datasets:
        try:
            path = download_2k_dataset(dataset_name, args.output_dir)
            print(f"Successfully downloaded {dataset_name} 2k dataset to {path}")
        except Exception as e:
            print(f"Error downloading {dataset_name} 2k dataset: {str(e)}")


if __name__ == "__main__":
    main()
