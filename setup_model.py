"""
Download SAM2 checkpoint before first run.
Usage:  python setup_model.py
"""

import os
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "tiny":  ("sam2.1_hiera_tiny.pt",
              "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
    "small": ("sam2.1_hiera_small.pt",
              "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"),
    "base":  ("sam2.1_hiera_base_plus.pt",
              "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"),
    "large": ("sam2.1_hiera_large.pt",
              "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"),
}

MODEL_CONFIGS = {
    "tiny":  "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def download_checkpoint(variant: str = "large", dest_dir: str = "checkpoints"):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    filename, url = CHECKPOINTS[variant]
    dest = os.path.join(dest_dir, filename)

    if os.path.exists(dest):
        print(f"✓ Checkpoint already exists: {dest}")
        return dest

    print(f"Downloading {variant} checkpoint …")
    urllib.request.urlretrieve(url, dest)
    print(f"✓ Saved to {dest}")
    return dest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=CHECKPOINTS.keys(), default="large")
    args = p.parse_args()
    download_checkpoint(args.variant)