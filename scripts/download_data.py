
import os
import requests
import tarfile
import zipfile
import io
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("data/satlib/raw")
URLS = {
    "rnd3sat": [
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf100-430.tar.gz",
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf200-860.tar.gz",
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz",
    ],
    "bms": [
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/BMS/RTI_k3_n100_m429.tar.gz"
    ],
    "cbs": [
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b10.tar.gz"
    ],
    "gcp": [
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/GCP/flat100-239.tar.gz"
    ],
    "sw_gcp": [
        "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/SW-GCP/sw100-8-lp0-lp8.tar.gz"
    ]
}

def download_and_extract(url, category):
    dest_dir = DATA_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {url} to {dest_dir}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_like_object = io.BytesIO(response.content)
        
        if url.endswith(".tar.gz"):
            with tarfile.open(fileobj=file_like_object, mode="r:gz") as tar:
                tar.extractall(path=dest_dir)
        elif url.endswith(".zip"):
             with zipfile.ZipFile(file_like_object) as z:
                z.extractall(path=dest_dir)
        
        print(f"Successfully extracted to {dest_dir}")
        
    except Exception as e:
        print(f"Failed to download/extract {url}: {e}")

def main():
    print(f"Downloading SATLIB datasets to {DATA_DIR}...")
    for category, urls in URLS.items():
        for url in urls:
            download_and_extract(url, category)
    print("Done!")

if __name__ == "__main__":
    main()
