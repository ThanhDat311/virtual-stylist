"""
scripts/visualize_embeddings.py
================================
Giai đoạn 1 — Milestone check: Trực quan hóa embedding clusters bằng T-SNE / PCA.

Mục đích:
  - Kiểm tra xem các sản phẩm cùng danh mục có "tụ lại thành cụm" không.
  - Nếu clusters rõ ràng → embeddings tốt → chuyển sang Giai đoạn 2.
  - Nếu clusters rối → cần kiểm tra lại model hoặc data.

Output:
  - outputs/tsne_visualization.png   (T-SNE 2D scatter)
  - outputs/pca_visualization.png    (PCA 2D scatter — nhanh hơn)
  - outputs/cluster_report.txt       (thống kê số item theo danh mục)

Chạy:
  python scripts/visualize_embeddings.py [--method tsne|pca|both]
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Color palette (per category) ─────────────────────────────────────────────
CATEGORY_COLORS = {
    "Apparel": "#4E79A7",
    "Footwear": "#F28E2B",
    "Accessories": "#59A14F",
    "Personal Care": "#E15759",
    "Free Items": "#B07AA1",
    "Unknown": "#9C755F",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data():
    """Load embeddings and category labels from metadata."""
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"❌  embeddings.npy not found at {EMBEDDINGS_PATH}")
        print("   Run  python scripts/prepare_data.py  first.")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Embeddings loaded: {embeddings.shape}")

    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Build parallel label arrays aligned with embeddings via faiss_index field
    n = embeddings.shape[0]
    categories = ["Unknown"] * n
    names = [""] * n

    for item in metadata.values():
        fi = item.get("faiss_index")
        if fi is not None and 0 <= fi < n:
            categories[fi] = item.get("category", "Unknown")
            names[fi] = item.get("name", "")

    return embeddings, categories, names


def reduce_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000):
    from sklearn.manifold import TSNE

    print(f"Running T-SNE (perplexity={perplexity}, n_iter={n_iter}) ...")
    t0 = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(embeddings) - 1),
        n_iter=n_iter,
        random_state=42,
        verbose=1,
    )
    coords = tsne.fit_transform(embeddings)
    print(f"  T-SNE done in {time.time() - t0:.1f}s")
    return coords


def reduce_pca(embeddings, n_components=2):
    from sklearn.decomposition import PCA

    print("Running PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    print(f"  PCA done. Explained variance: PC1={var[0]*100:.1f}% PC2={var[1]*100:.1f}%")
    return coords, var


def plot_scatter(coords, categories, title, output_path, var_ratio=None):
    """Draw a styled 2D scatter plot coloured by category."""
    unique_cats = sorted(set(categories))
    palette = {c: CATEGORY_COLORS.get(c, "#76B7B2") for c in unique_cats}

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        color = palette[cat]
        ax.scatter(
            xs, ys,
            c=color,
            s=18,
            alpha=0.75,
            linewidths=0,
            label=f"{cat} (n={mask.count(True)})",
        )

    # Title & labels
    ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=16)
    xlabel = "Component 1"
    ylabel = "Component 2"
    if var_ratio is not None:
        xlabel += f"  ({var_ratio[0]*100:.1f}% var)"
        ylabel += f"  ({var_ratio[1]*100:.1f}% var)"
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=11)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=11)
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Legend
    legend = ax.legend(
        framealpha=0.25, facecolor="#0f3460", edgecolor="#333355",
        labelcolor="white", fontsize=10, markerscale=1.8,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {output_path}")


def write_cluster_report(categories, names):
    """Write a simple text report of category distribution."""
    from collections import Counter

    report_path = os.path.join(OUTPUTS_DIR, "cluster_report.txt")
    counts = Counter(categories)
    lines = [
        "=== Virtual Stylist — Embedding Cluster Report ===\n",
        f"Total items in FAISS index: {len(categories)}\n\n",
        "Category distribution:\n",
    ]
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // max(1, len(categories) // 40))
        lines.append(f"  {cat:<20} {cnt:>4}  {bar}\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nCluster report → {report_path}")

    # Print to console too
    print("\n" + "".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main(method: str, max_samples: int):
    print("=" * 60)
    print("  Virtual Stylist — Embedding Visualizer")
    print("=" * 60)

    embeddings, categories, names = load_data()

    # Optionally subsample for speed
    if max_samples > 0 and len(embeddings) > max_samples:
        idx = np.random.default_rng(42).choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        categories = [categories[i] for i in idx]
        names = [names[i] for i in idx]
        print(f"Subsampled to {max_samples} points for visualisation.")

    write_cluster_report(categories, names)

    if method in ("pca", "both"):
        coords_pca, var = reduce_pca(embeddings)
        plot_scatter(
            coords_pca, categories,
            title="Fashion Embeddings — PCA (Giai đoạn 1 Milestone Check)",
            output_path=os.path.join(OUTPUTS_DIR, "pca_visualization.png"),
            var_ratio=var,
        )

    if method in ("tsne", "both"):
        # For T-SNE, subsample more aggressively if > 5000
        emb_tsne = embeddings
        cats_tsne = categories
        if len(embeddings) > 5000:
            idx = np.random.default_rng(42).choice(len(embeddings), 5000, replace=False)
            emb_tsne = embeddings[idx]
            cats_tsne = [categories[i] for i in idx]
            print(f"T-SNE: subsampled to 5000 for speed.")

        coords_tsne = reduce_tsne(emb_tsne)
        plot_scatter(
            coords_tsne, cats_tsne,
            title="Fashion Embeddings — T-SNE (Giai đoạn 1 Milestone Check)",
            output_path=os.path.join(OUTPUTS_DIR, "tsne_visualization.png"),
        )

    print("\n✅ Visualization hoàn thành!")
    print("   Kiểm tra outputs/ để xem kết quả:")
    plots = []
    if method in ("pca", "both"):
        plots.append("  • outputs/pca_visualization.png")
    if method in ("tsne", "both"):
        plots.append("  • outputs/tsne_visualization.png")
    plots.append("  • outputs/cluster_report.txt")
    print("\n".join(plots))
    print("\n📌  Nếu các cụm theo danh mục rõ ràng → Embeddings ổn, chuyển sang Giai đoạn 2!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise CLIP embeddings via T-SNE or PCA")
    parser.add_argument(
        "--method", choices=["tsne", "pca", "both"], default="pca",
        help="Reduction method (pca is fast, tsne is more accurate but slow).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Max number of embeddings to visualise (0 = all).",
    )
    args = parser.parse_args()
    main(args.method, args.max_samples)
