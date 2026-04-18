"""
scripts/prepare_data.py
=======================
Giai đoạn 1: Chuẩn bị dữ liệu và trích xuất embeddings.

Quy trình:
  1. Tải dataset từ Kaggle bằng kagglehub
  2. Lọc và resize ảnh về 224×224, lưu vào data/processed/
  3. Trích xuất CLIP embeddings
  4. Lưu embeddings.npy + image_paths.npy
  5. Build FAISS index (data/vector_index.bin) và cập nhật metadata.json

Chạy:
  python scripts/prepare_data.py [--max_images 1000] [--batch_size 32]
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.feature_extraction import FeatureExtractor

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
IMAGE_PATHS_FILE = os.path.join(DATA_DIR, "image_paths.npy")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "vector_index.bin")

# ── Helpers ──────────────────────────────────────────────────────────────────

def download_kaggle_dataset():
    """Download the fashion-product-images-dataset via kagglehub."""
    print("\n[1/5] Downloading dataset from Kaggle...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(
            "paramaggarwal/fashion-product-images-dataset"
        )
        print(f"  Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  ❌ kagglehub download failed: {e}")
        print("  → Make sure you have set KAGGLE_USERNAME and KAGGLE_KEY env vars.")
        return None


def find_images_root(kaggle_root):
    """
    Walk the downloaded directory to find the 'images' subfolder.
    Returns the first dir that contains .jpg files.
    """
    for dirpath, dirnames, filenames in os.walk(kaggle_root):
        jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
        if jpgs:
            return dirpath
    return None


def load_metadata():
    """Load existing metadata.json → dict keyed by product-id string."""
    if not os.path.exists(METADATA_PATH):
        print(f"  ❌ metadata.json not found at {METADATA_PATH}")
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"  Loaded metadata: {len(meta)} items")
    return meta


def validate_and_resize_image(src_path, dst_path, size=(224, 224)):
    """
    Open image at src_path, resize to size, save to dst_path.
    Returns True on success, False on error.
    """
    try:
        with Image.open(src_path) as img:
            # Skip corrupt / tiny images
            if img.size[0] < 32 or img.size[1] < 32:
                return False
            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize(size, Image.LANCZOS)
            img_resized.save(dst_path, format="JPEG", quality=90)
        return True
    except (UnidentifiedImageError, OSError, Exception):
        return False


def build_faiss_index(embeddings, index_path):
    """Build a flat L2 FAISS index and save to disk."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    print(f"  FAISS index: {index.ntotal} vectors, dim={dim} → saved to {index_path}")
    return index


# ── Main ─────────────────────────────────────────────────────────────────────

def main(max_images: int, batch_size: int, skip_download: bool):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "outputs"), exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if skip_download:
        print("[1/5] Skipping Kaggle download (--skip_download)")
        kaggle_root = None
    else:
        kaggle_root = download_kaggle_dataset()

    images_root = None
    if kaggle_root:
        images_root = find_images_root(kaggle_root)
        if images_root:
            print(f"  Found images folder: {images_root}")
        else:
            print("  ⚠️  Could not find images folder inside downloaded dataset.")

    # ── Step 2: Load metadata & build local file list ─────────────────────────
    print("\n[2/5] Loading metadata and locating images...")
    metadata = load_metadata()

    # Build list of (meta_index, product_id, source_image_path)
    items = []
    for idx, item in metadata.items():
        product_id = item.get("id", "")
        local_processed = os.path.join(PROCESSED_DIR, f"{product_id}.jpg")

        # Check if already processed
        if os.path.exists(local_processed):
            items.append((idx, product_id, local_processed))
            continue

        # Try to find the source image
        kaggle_path = item.get("image_path", "")
        # Try the original kaggle path (works if running on Colab)
        if os.path.exists(kaggle_path):
            items.append((idx, product_id, kaggle_path))
        elif images_root:
            # Construct local path from downloaded dataset
            filename = os.path.basename(kaggle_path)
            candidate = os.path.join(images_root, filename)
            if os.path.exists(candidate):
                items.append((idx, product_id, candidate))
            else:
                # Skip: image not found
                pass
        # else: no source found, skip silently

    if max_images > 0:
        items = items[:max_images]

    print(f"  Items to process: {len(items)}")

    # ── Step 3: Resize & clean ────────────────────────────────────────────────
    print(f"\n[3/5] Filtering & resizing images → {PROCESSED_DIR}")
    valid_items = []
    skipped = 0

    for idx, product_id, src_path in tqdm(items, desc="Resizing", unit="img"):
        dst_path = os.path.join(PROCESSED_DIR, f"{product_id}.jpg")
        if os.path.exists(dst_path):
            valid_items.append((idx, product_id, dst_path))
            continue
        ok = validate_and_resize_image(src_path, dst_path)
        if ok:
            valid_items.append((idx, product_id, dst_path))
        else:
            skipped += 1

    print(f"  ✅ Valid: {len(valid_items)}  |  ❌ Skipped (corrupt/tiny): {skipped}")

    if not valid_items:
        print("  No valid images found. Exiting.")
        sys.exit(1)

    # ── Step 4: Extract CLIP embeddings ───────────────────────────────────────
    print("\n[4/5] Extracting CLIP feature embeddings...")
    extractor = FeatureExtractor()

    # Sort to keep deterministic order
    valid_items.sort(key=lambda x: x[0])
    meta_indices = [x[0] for x in valid_items]
    product_ids = [x[1] for x in valid_items]
    image_paths = [x[2] for x in valid_items]

    embeddings = extractor.extract_features_batch(
        image_paths, batch_size=batch_size, verbose=True
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save embeddings + image path mapping
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(IMAGE_PATHS_FILE, np.array(image_paths))
    print(f"  Saved: {EMBEDDINGS_PATH}")
    print(f"  Saved: {IMAGE_PATHS_FILE}")

    # ── Step 5: Build FAISS index & update metadata ───────────────────────────
    print("\n[5/5] Building FAISS index & updating metadata...")
    build_faiss_index(embeddings, FAISS_INDEX_PATH)

    # Update metadata.json with local processed paths and faiss_index
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        full_meta = json.load(f)

    for faiss_idx, (meta_idx, product_id, local_path) in enumerate(
        zip(meta_indices, product_ids, image_paths)
    ):
        if meta_idx in full_meta:
            full_meta[meta_idx]["local_image_path"] = local_path
            full_meta[meta_idx]["faiss_index"] = faiss_idx

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, ensure_ascii=False, indent=None)

    print(f"  metadata.json updated with local_image_path and faiss_index fields")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Giai đoạn 1 hoàn thành!")
    print(f"   Items in FAISS index : {embeddings.shape[0]}")
    print(f"   Embedding dimension  : {embeddings.shape[1]}")
    print(f"   Processed images     : {PROCESSED_DIR}")
    print(f"   Embeddings file      : {EMBEDDINGS_PATH}")
    print(f"   FAISS index          : {FAISS_INDEX_PATH}")
    print("=" * 60)
    print("\nBước tiếp theo: chạy  python scripts/visualize_embeddings.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fashion dataset for Virtual Stylist")
    parser.add_argument(
        "--max_images", type=int, default=0,
        help="Limit number of images (0 = no limit). Useful for quick testing.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for CLIP feature extraction.",
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip Kaggle download (if images already cached).",
    )
    args = parser.parse_args()
    main(args.max_images, args.batch_size, args.skip_download)
