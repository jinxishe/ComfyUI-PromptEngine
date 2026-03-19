"""
step2_clustering.py - Step 2 聚类分析函数

节点版实现以根目录 step2_full_pipeline_with_summary.py 为准，
保持聚类流程、默认参数与统计输出逻辑一致。
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import hdbscan
import torch
import umap
from sentence_transformers import SentenceTransformer

from .step1_dimension_extract import DIMS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR_NAME = "bge-small-en-v1.5"
MODEL_REPO_ID = "BAAI/bge-small-en-v1.5"


def get_model_search_paths() -> list[Path]:
    """Return local model search paths ordered by priority."""
    current_file = Path(__file__).resolve()
    plugin_dir = current_file.parent.parent
    comfyui_dir = plugin_dir.parent.parent

    return [
        comfyui_dir / "models" / "embeddings" / MODEL_DIR_NAME,
        comfyui_dir / "models" / "embedding" / MODEL_DIR_NAME,
        comfyui_dir / "models" / "sentence_transformers" / MODEL_DIR_NAME,
        plugin_dir / "model" / MODEL_DIR_NAME,
    ]


def get_preferred_model_dir() -> Path:
    return get_model_search_paths()[0]


def download_model(target_dir: Path) -> Path:
    """Download the model from Hugging Face on first use if missing locally."""
    print(f"Local model not found. Downloading from Hugging Face: {MODEL_REPO_ID}")
    print(f"Target directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic model download. Please install sentence-transformers or huggingface_hub."
        ) from exc

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Model download completed: {target_dir}")
    return target_dir


def resolve_model_path() -> Path:
    for path in get_model_search_paths():
        if path.exists():
            return path

    return download_model(get_preferred_model_dir())


def load_model():
    """Load the local embedding model, preferring the ComfyUI models directory."""
    print(f"Using device: {DEVICE.upper()}")
    print("Loading local model...")
    model_path = resolve_model_path()
    try:
        model = SentenceTransformer(
            str(model_path),
            device=DEVICE,
            local_files_only=True,
            trust_remote_code=False,
        )
        print(f"Loaded local model from: {model_path}")
        return model
    except Exception as e:
        print(f"Local model load failed: {e}")
        print("Please make sure the directory is valid and contains files such as model.safetensors, config.json, and tokenizer assets.")
        raise


def get_cluster_output_dir() -> Path:
    current_file = Path(__file__).resolve()
    plugin_dir = current_file.parent.parent
    return plugin_dir / "output" / "step2_clusters"


def build_summary_markdown(results: dict, output_dir: Path) -> str:
    total_dims = len(results)
    total_clusters = sum(r["main_clusters"] + r["sub_clusters"] for r in results.values())
    lines = [
        "# Step 2 Completed",
        "",
        f"- Dimensions processed: {total_dims}",
        f"- Total clusters: {total_clusters}",
        f"- Cluster directory: `{output_dir}`",
        "",
        "| Dimension | Unique Values | Main Clusters | Sub Clusters | Main Noise | Sub Noise | Residual Noise | Time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dim, stats in results.items():
        lines.append(
            f"| {dim} | {stats['unique_count']} | {stats['main_clusters']} | "
            f"{stats['sub_clusters']} | {stats['main_noise_count']} | "
            f"{stats['sub_noise_count']} | {stats['residual_noise_count']} | "
            f"{stats['duration_seconds']:.1f} |"
        )

    return "\n".join(lines)


def cluster_once(model, texts, min_size, min_samples, method, name_prefix=""):
    """Run one HDBSCAN clustering pass, matching the original script behavior."""
    start_time = time.time()

    if len(texts) == 0:
        return {}, 0

    if len(texts) < min_size * 2:
        print(f"  Dataset too small ({len(texts)} items); using a single cluster directly")
        clusters = {f"{name_prefix}single": texts}
        duration = time.time() - start_time
        print(f"  Time: {duration:.1f}s")
        return clusters, 0

    print(f"  Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("  Running UMAP...")
    n_texts = len(texts)
    n_neighbors = min(15, max(2, n_texts - 1))
    # Keep n_components safely below N - 1 on small datasets to avoid
    # spectral initialization failures during the noise reclustering pass.
    n_components = min(10, max(2, n_texts - 2))
    umap_init = "random" if n_texts <= 12 else "spectral"

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
        init=umap_init,
    )
    emb_reduced = reducer.fit_transform(embeddings)

    print(f"  Running HDBSCAN ({method}, min_size={min_size}, min_samples={min_samples})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=method,
    )
    labels = clusterer.fit_predict(emb_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(list(labels).count(-1))

    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl == -1:
            clusters["noise"].append(texts[i])
        else:
            clusters[f"{name_prefix}cluster_{lbl}"].append(texts[i])

    duration = time.time() - start_time
    print(f"  -> {n_clusters} clusters, {n_noise} noise items ({n_noise/len(texts):.1%}), time {duration:.1f}s")
    return dict(clusters), n_noise


def run_clustering_pipeline(input_path, enable_noise_recluster=True,
                           primary_min_cluster_size=6,
                           primary_min_samples=3,
                           noise_min_cluster_size=4,
                           noise_min_samples=2):
    """
    Complete clustering pipeline wrapper.

    This implementation matches step2_full_pipeline_with_summary.py:
    - Same small-sample single-cluster behavior
    - Same UMAP / HDBSCAN configuration
    - Same noise reclustering trigger rule (len(noise_items) >= 10)
    """
    primary_method = "leaf"
    noise_method = "leaf"

    output_base_dir = get_cluster_output_dir()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading JSON file: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    records = [(pid, item) for pid, item in raw_data.items() if "_error" not in item]
    print(f"Valid records: {len(records)}\n")

    model = load_model()
    results = {}

    for dim in DIMS:
        print(f"\n{'='*80}")
        print(f"Processing dimension: {dim}")
        print(f"{'='*80}")

        dim_start_time = time.time()
        values = [item.get(dim, "").strip() for _, item in records if item.get(dim, "").strip()]
        if not values:
            print("  No valid values in this dimension. Skipping.")
            continue

        unique_values = list(set(values))
        unique_count = len(unique_values)
        print(f"  Unique values: {unique_count}")

        main_clusters, main_noise_count = cluster_once(
            model,
            unique_values,
            min_size=primary_min_cluster_size,
            min_samples=primary_min_samples,
            method=primary_method,
            name_prefix="main_",
        )

        main_noise_rate = main_noise_count / unique_count if unique_count > 0 else 0
        noise_items = main_clusters.pop("noise", [])
        sub_clusters_count = 0
        sub_noise_count = 0
        sub_noise_rate = 0

        if enable_noise_recluster and len(noise_items) >= 10:
            print(f"\n  Reclustering noise items ({len(noise_items)} items)...")
            noise_clusters, sub_noise_count = cluster_once(
                model,
                noise_items,
                min_size=noise_min_cluster_size,
                min_samples=noise_min_samples,
                method=noise_method,
                name_prefix="sub_",
            )

            sub_clusters_count = len([k for k in noise_clusters if k != "noise"])

            for sub_key, sub_items in noise_clusters.items():
                if sub_key == "noise":
                    main_clusters.setdefault("residual_noise", []).extend(sub_items)
                else:
                    main_clusters[sub_key] = sub_items

            sub_noise_rate = sub_noise_count / len(noise_items) if len(noise_items) > 0 else 0
        elif noise_items:
            main_clusters["residual_noise"] = noise_items

        residual_noise = main_clusters.get("residual_noise", [])
        residual_noise_count = len(residual_noise)
        total_processed = sum(len(v) for v in main_clusters.values())
        residual_noise_rate = residual_noise_count / total_processed if total_processed > 0 else 0
        duration = time.time() - dim_start_time

        dim_dir = output_base_dir / dim
        dim_dir.mkdir(exist_ok=True)
        output_file = dim_dir / f"{dim}_clusters.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dict(main_clusters), f, ensure_ascii=False, indent=2)

        print(f"  Saved result: {output_file}")

        cluster_sizes = {k: len(v) for k, v in main_clusters.items() if k != "residual_noise"}
        sorted_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]

        print("\n  Main cluster sizes (top 10):")
        for k, cnt in sorted_sizes:
            print(f"    {k:18} : {cnt:4d} items")

        print(f"\n  Processed {total_processed} items, residual noise {residual_noise_count} ({residual_noise_rate:.1%}), time {duration:.1f}s\n")

        results[dim] = {
            "unique_count": unique_count,
            "main_clusters": len([k for k in main_clusters if not k.startswith("sub_") and k != "residual_noise" and k != "noise"]),
            "sub_clusters": sub_clusters_count,
            "main_noise_count": main_noise_count,
            "sub_noise_count": sub_noise_count,
            "residual_noise_count": residual_noise_count,
            "duration_seconds": duration,
        }

    summary_markdown = build_summary_markdown(results, output_base_dir.resolve())
    print("\nAll dimensions processed.")
    print(f"Output directory: {output_base_dir.resolve()}")
    return results, summary_markdown, str(output_base_dir.resolve())


if __name__ == "__main__":
    test_json = "data/02_merged_normal_20260211_001-026.json"
    results, summary_markdown, output_dir = run_clustering_pipeline(test_json)
    print(summary_markdown)
    print(f"\nOutput directory: {output_dir}")
    print("\nClustering summary:")
    for dim, stats in results.items():
        print(f"  {dim}: {stats}")
