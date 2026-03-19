"""
step3_dictionary_gen.py — Step 3 词典生成函数

供 ComfyUI 节点调用，支持全量/增量/单维度模式
"""

import json
import re
import asyncio
import threading
from pathlib import Path
from datetime import date
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

from .step1_dimension_extract import DIMS

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

CONFIG = {
    "base_url":        "http://localhost:8080/v1",
    "model":           "Qwen3.5-35B-A3B-heretic-v2-mxfp4_moe",
    "concurrency":     4,
    "max_tokens":      400,
    "temperature":     0.1,
    "input_dir":       Path(__file__).resolve().parent.parent / "output" / "step2_clusters",
    "output_dir":      Path(__file__).resolve().parent.parent / "output" / "step3_dictionaries",
    "version":         "1.0",
}

# 枚举直通维度
ENUM_DIMS = {"shot_distance", "ethnicity", "hair_color", "age_appearance"}
FLATTEN_DIMS = {"hair_color"}


def normalize_enum_key(text: str) -> str:
    text = (text or "").strip().lower().replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def run_async_compatible(coro):
    """Run a coroutine safely inside environments that may already own an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box = {}
    error_box = {}

    def runner():
        try:
            result_box["value"] = asyncio.run(coro)
        except Exception as exc:
            error_box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


# ─────────────────────────────────────────────
# LLM 调用函数
# ─────────────────────────────────────────────

async def call_llm(client, prompt):
    """调用本地 LLM"""
    response = await client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": "You are a precise T2I prompt naming assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"]
    )
    return response.choices[0].message.content.strip()


def build_enum_alias_map(*dict_layers: dict | None) -> dict[str, tuple[str, dict]]:
    alias_map = {}
    for dictionary in dict_layers:
        for key, cluster in (dictionary or {}).get("clusters", {}).items():
            aliases = {
                key,
                cluster.get("canonical_name", ""),
                cluster.get("canonical_name_zh", ""),
                cluster.get("canonical_phrase", ""),
                *cluster.get("samples", []),
            }
            for alias in aliases:
                normalized = normalize_enum_key(alias)
                if normalized:
                    alias_map[normalized] = (key, cluster)
    return alias_map


def generate_enum_dictionary(dimension, clusters_data, base_dict=None, user_dict=None):
    """
    为枚举直通维度生成词典
    
    Args:
        dimension: 维度名
        clusters_data: 聚类数据
        
    Returns:
        dict: 词典结构
    """
    clusters = {}
    alias_map = build_enum_alias_map(base_dict, user_dict)

    for cluster_name, samples in clusters_data.items():
        if cluster_name == "residual_noise":
            continue

        for sample in dedupe_preserve_order(samples):
            normalized_sample = normalize_enum_key(sample)
            if not normalized_sample:
                continue

            existing = alias_map.get(normalized_sample)
            if existing:
                key, existing_cluster = existing
                cluster = clusters.setdefault(key, {
                    "canonical_name": existing_cluster.get("canonical_name", sample.title()),
                    "canonical_name_zh": existing_cluster.get("canonical_name_zh", sample),
                    "canonical_phrase": existing_cluster.get("canonical_phrase", sample),
                    "samples": [],
                    "tags": list(existing_cluster.get("tags", [])),
                })
            else:
                key = normalized_sample
                cluster = clusters.setdefault(key, {
                    "canonical_name": sample.title(),
                    "canonical_name_zh": sample,
                    "canonical_phrase": sample,
                    "samples": [],
                    "tags": dedupe_preserve_order(
                        [token.lower() for token in re.split(r"[\s,_\-]+", sample) if token][:5]
                    ),
                })

            cluster["samples"] = dedupe_preserve_order(cluster["samples"] + [sample])
    
    return {
        "dimension": dimension,
        "version": CONFIG["version"],
        "generated_date": str(date.today()),
        "clusters": clusters,
        "statistics": {
            "total_clusters": len(clusters),
            "multi_sample_clusters": sum(1 for c in clusters.values() if len(c["samples"]) > 1),
            "single_sample_clusters": sum(1 for c in clusters.values() if len(c["samples"]) == 1)
        }
    }


def dedupe_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def load_dictionary_map(directory: Path, dimensions: list[str]) -> dict[str, dict]:
    loaded = {}
    if not directory.exists():
        return loaded

    for dim in dimensions:
        dict_file = directory / f"{dim}_dict.json"
        if dict_file.exists():
            with open(dict_file, "r", encoding="utf-8") as f:
                loaded[dim] = json.load(f)
    return loaded


def build_incremental_dictionary(dimension: str, generated_dict: dict, base_dict: dict, user_dict: dict) -> dict | None:
    """Build the incremental-only dictionary payload for one dimension."""
    base_clusters = (base_dict or {}).get("clusters", {})
    user_clusters = (user_dict or {}).get("clusters", {})
    incremental_clusters = {}

    for key, cluster in generated_dict.get("clusters", {}).items():
        if key not in base_clusters and key not in user_clusters:
            new_cluster = dict(cluster)
            new_cluster["samples"] = dedupe_preserve_order(cluster.get("samples", []))
            incremental_clusters[key] = new_cluster
            continue

        existing_cluster = base_clusters.get(key) or user_clusters.get(key) or {}
        existing_samples = set(base_clusters.get(key, {}).get("samples", []))
        existing_samples.update(user_clusters.get(key, {}).get("samples", []))
        new_samples = [
            sample for sample in cluster.get("samples", [])
            if sample not in existing_samples
        ]
        if not new_samples:
            continue

        incremental_clusters[key] = {
            "canonical_name": existing_cluster.get("canonical_name", cluster.get("canonical_name", "")),
            "canonical_name_zh": existing_cluster.get("canonical_name_zh", cluster.get("canonical_name_zh", "")),
            "canonical_phrase": existing_cluster.get("canonical_phrase", cluster.get("canonical_phrase", "")),
            "samples": dedupe_preserve_order(new_samples),
            "tags": list(existing_cluster.get("tags", cluster.get("tags", []))),
        }

    if not incremental_clusters:
        return None

    return {
        "dimension": dimension,
        "version": generated_dict.get("version", CONFIG["version"]),
        "generated_date": str(date.today()),
        "mode": "incremental_patch",
        "clusters": incremental_clusters,
        "statistics": {
            "total_clusters": len(incremental_clusters),
            "new_keys": sum(1 for key in incremental_clusters if key not in base_clusters and key not in user_clusters),
            "updated_keys": sum(1 for key in incremental_clusters if key in base_clusters or key in user_clusters),
            "new_samples": sum(len(cluster.get("samples", [])) for cluster in incremental_clusters.values()),
        }
    }


def merge_into_user_dictionary(existing_user_dict: dict | None, incremental_dict: dict) -> dict:
    """Merge a new incremental patch into the persisted user dictionary."""
    if not existing_user_dict:
        return incremental_dict

    merged = dict(existing_user_dict)
    merged_clusters = {key: dict(cluster) for key, cluster in existing_user_dict.get("clusters", {}).items()}

    for key, cluster in incremental_dict.get("clusters", {}).items():
        if key in merged_clusters:
            existing_samples = merged_clusters[key].get("samples", [])
            merged_clusters[key]["samples"] = dedupe_preserve_order(existing_samples + cluster.get("samples", []))
        else:
            merged_clusters[key] = dict(cluster)
            merged_clusters[key]["samples"] = dedupe_preserve_order(cluster.get("samples", []))

    merged["clusters"] = merged_clusters
    merged["generated_date"] = str(date.today())
    merged["mode"] = "incremental_patch"
    merged["statistics"] = {
        "total_clusters": len(merged_clusters),
        "total_samples": sum(len(cluster.get("samples", [])) for cluster in merged_clusters.values()),
    }
    return merged


async def generate_llm_dictionary(client, dimension, clusters_data, test_mode=False):
    """
    使用 LLM 为开放维度生成词典
    
    Args:
        client: AsyncOpenAI 客户端
        dimension: 维度名
        clusters_data: 聚类数据
        test_mode: 是否仅测试前 3 个簇
        
    Returns:
        dict: 词典结构
    """
    clusters = {}
    processed_keys = set()
    
    # 准备任务列表
    tasks = []
    cluster_items = list(clusters_data.items())
    
    if test_mode:
        cluster_items = cluster_items[:3]
    
    for cluster_name, samples in cluster_items:
        if cluster_name == "residual_noise":
            continue
            
        # 构建 Prompt
        is_single_sample = len(samples) == 1
        
        prompt = f"""You are naming clusters of T2I prompt phrases for the '{dimension}' dimension.

Cluster samples:
{chr(10).join(f'- {s}' for s in samples)}

Tasks:
1. canonical_name: A concise English name (3-6 words) capturing the shared features of ALL samples. {'This cluster has only one sample — describe its core features directly, be specific.' if is_single_sample else 'Extract the greatest common denominator — ignore individual details.'}
2. canonical_name_zh: Chinese translation of canonical_name (4-10 characters).
3. canonical_phrase: A short English phrase (5-12 words) suitable for direct use in a T2I prompt. {'Use the sample text as-is, do not paraphrase.' if is_single_sample else 'More specific than canonical_name but still representative of the cluster.'}
4. tags: Exactly 5 keywords that users might SEARCH FOR. Cover: color, material, garment type, style vibe, one unique detail. All lowercase, single words preferred.

Return exactly this JSON and nothing else:
{{
  "canonical_name": "",
  "canonical_name_zh": "",
  "canonical_phrase": "",
  "tags": []
}}"""
        
        tasks.append((cluster_name, samples, prompt))
    
    # 并发调用 LLM
    semaphore = asyncio.Semaphore(CONFIG["concurrency"])
    
    async def process_task(cluster_name, samples, prompt):
        async with semaphore:
            try:
                response = await call_llm(client, prompt)
                # 解析 JSON
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    
                    # 生成 key
                    key = result.get("canonical_name", cluster_name).lower().replace(" ", "_").replace("-", "_")
                    key = re.sub(r'[^a-z0-9_]', '', key)
                    
                    # 去重
                    original_key = key
                    counter = 2
                    while key in processed_keys:
                        key = f"{original_key}_{counter}"
                        counter += 1
                    
                    processed_keys.add(key)
                    
                    return key, result, samples
                else:
                    print(f"Warning: failed to parse JSON for cluster {cluster_name}")
                    return None
            except Exception as e:
                print(f"Error: LLM call failed for {cluster_name}: {e}")
                return None
    
    results = await atqdm.gather(
        *[process_task(name, samples, prompt) for name, samples, prompt in tasks],
        desc=f"LLM processing {dimension}"
    )
    
    # 整理结果
    for result in results:
        if result:
            key, llm_data, samples = result
            clusters[key] = {
                "canonical_name": llm_data.get("canonical_name", ""),
                "canonical_name_zh": llm_data.get("canonical_name_zh", ""),
                "canonical_phrase": llm_data.get("canonical_phrase", ""),
                "samples": samples,
                "tags": llm_data.get("tags", [])
            }
    
    return {
        "dimension": dimension,
        "version": CONFIG["version"],
        "generated_date": str(date.today()),
        "clusters": clusters,
        "statistics": {
            "total_clusters": len(clusters),
            "multi_sample_clusters": sum(1 for c in clusters.values() if len(c["samples"]) > 1),
            "single_sample_clusters": sum(1 for c in clusters.values() if len(c["samples"]) == 1)
        }
    }


def run_dictionary_generation(base_dict_dir="dim_dictionaries", input_dir=None, llm_config=None):
    """
    Complete incremental dictionary generation pipeline wrapper
    
    Args:
        base_dict_dir: Base dictionary directory
        input_dir: Cluster result directory (optional)
        llm_config: Optional LLM config dict from the node input
        
    Returns:
        list: Updated dimensions
    """
    global CONFIG

    if input_dir:
        CONFIG["input_dir"] = Path(input_dir)
    if llm_config:
        CONFIG.update({
            "base_url": llm_config.get("base_url", CONFIG["base_url"]),
            "model": llm_config.get("model", CONFIG["model"]),
            "concurrency": llm_config.get("concurrency", CONFIG["concurrency"]),
            "max_tokens": llm_config.get("max_tokens", CONFIG["max_tokens"]),
            "temperature": llm_config.get("temperature", CONFIG["temperature"]),
        })
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    dimensions_to_process = DIMS
    
    base_path = Path(base_dict_dir)
    print(f"\nLoading base dictionaries from: {base_path}")
    base_dicts = load_dictionary_map(base_path, dimensions_to_process)
    print(f"Loaded base dictionaries for {len(base_dicts)} dimensions")

    print(f"Loading existing user dictionaries from: {CONFIG['output_dir']}")
    user_dicts = load_dictionary_map(CONFIG["output_dir"], dimensions_to_process)
    print(f"Loaded existing user dictionaries for {len(user_dicts)} dimensions")
    
    # 初始化 LLM 客户端
    client = AsyncOpenAI(
        base_url=CONFIG["base_url"],
        api_key=llm_config.get("api_key", "not-needed") if llm_config else "not-needed"
    )
    
    processed = []
    
    for dim in dimensions_to_process:
        print(f"\n{'='*50}")
        print(f"[{dim}] Processing started")
        print(f"{'='*50}")
        
        # 加载聚类数据
        cluster_file = CONFIG["input_dir"] / dim / f"{dim}_clusters.json"
        if not cluster_file.exists():
            print(f"⚠️  Skipped: cluster file not found: {cluster_file}")
            continue
        
        with open(cluster_file, 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
        
        if dim in ENUM_DIMS:
            print("  Using enum passthrough mode")
            dictionary = generate_enum_dictionary(
                dim,
                clusters_data,
                base_dict=base_dicts.get(dim, {}),
                user_dict=user_dicts.get(dim, {}),
            )
        else:
            print("  Using LLM naming mode")
            dictionary = run_async_compatible(generate_llm_dictionary(
                client, dim, clusters_data, test_mode=False
            ))

        incremental_dict = build_incremental_dictionary(
            dim,
            dictionary,
            base_dicts.get(dim, {}),
            user_dicts.get(dim, {}),
        )

        if not incremental_dict:
            print("  No incremental updates detected. Nothing to save.")
            continue

        output_file = CONFIG["output_dir"] / f"{dim}_dict.json"
        merged_user_dict = merge_into_user_dictionary(user_dicts.get(dim), incremental_dict)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_user_dict, f, ensure_ascii=False, indent=2)

        print(f"  Completed. Saved incremental dictionary: {output_file}")
        print(f"  New keys: {incremental_dict['statistics']['new_keys']}")
        print(f"  Updated keys: {incremental_dict['statistics']['updated_keys']}")
        print(f"  New samples: {incremental_dict['statistics']['new_samples']}")

        user_dicts[dim] = merged_user_dict
        processed.append(dim)
    
    print("\nAll dimensions processed.")
    print(f"Processed dimensions: {len(processed)}")
    print(f"Output directory: {CONFIG['output_dir']}")
    
    return processed


if __name__ == "__main__":
    # 测试示例
    processed = run_dictionary_generation()
    print(f"\nProcessed dimensions: {processed}")
