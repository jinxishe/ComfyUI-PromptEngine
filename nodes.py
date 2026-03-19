"""
nodes.py — ComfyUI-PromptEngine 核心节点实现

节点：
  - PromptEngineNode : 单维度节点，通过 prompt_in 串联
  - PromptEngineFull : 全维度节点，21 个维度全部展开
  - Step1DimensionExtract : Step 1 维度提取节点
  - Step2Clustering : Step 2 聚类分析节点
  - Step3DictionaryGen : Step 3 词典生成节点
"""

import json
import random
import re
from pathlib import Path

# ─────────────────────────────────────────────
# 维度定义（与 Step 1 保持完全一致）
# ─────────────────────────────────────────────

DIMS = [
    "ethnicity", "gender", "age_appearance", "subject_appearance",
    "hair_style", "hair_color",
    "outfit", "accessories",
    "pose", "body_direction",
    "expression", "gaze",
    "location_type", "background_props", "atmosphere",
    "shot_angle", "shot_distance", "composition",
    "lighting", "color_grade", "visual_style",
]

# 性别维度（硬编码，无需词典）
GENDER_OPTIONS = ["man", "woman"]

# 维度显示名称（英文 / 中文）
DIM_DISPLAY = {
    "ethnicity":          {"en": "Ethnicity",       "zh": "种族"},
    "gender":             {"en": "Gender",          "zh": "性别"},  # 特殊：与 ethnicity 拼接输出
    "age_appearance":     {"en": "Age",              "zh": "年龄"},
    "subject_appearance": {"en": "Appearance",       "zh": "外貌"},
    "hair_style":         {"en": "Hair Style",       "zh": "发型"},
    "hair_color":         {"en": "Hair Color",       "zh": "发色"},
    "outfit":             {"en": "Outfit",           "zh": "服装"},
    "accessories":        {"en": "Accessories",      "zh": "配饰"},
    "pose":               {"en": "Pose",             "zh": "姿势"},
    "body_direction":     {"en": "Body Direction",   "zh": "朝向"},
    "expression":         {"en": "Expression",       "zh": "表情"},
    "gaze":               {"en": "Gaze",             "zh": "视线"},
    "location_type":      {"en": "Location",         "zh": "场景"},
    "background_props":   {"en": "Background",       "zh": "背景"},
    "atmosphere":         {"en": "Atmosphere",       "zh": "氛围"},
    "shot_angle":         {"en": "Shot Angle",       "zh": "角度"},
    "shot_distance":      {"en": "Shot Distance",    "zh": "景别"},
    "composition":        {"en": "Composition",      "zh": "构图"},
    "lighting":           {"en": "Lighting",         "zh": "光线"},
    "color_grade":        {"en": "Color Grade",      "zh": "色调"},
    "visual_style":       {"en": "Visual Style",     "zh": "风格"},
}

# 特殊 Style 选项
STYLE_RANDOM  = "🎲 Random Style"
STYLE_SKIP    = "── (skip) ──"

# 注意：control_after_generate 是 ComfyUI 内置保留参数，
# 不在 INPUT_TYPES 里声明，框架会自动为 seed 类型的 widget 附加它。
# 在 generate() 里不需要接收此参数。

# ─────────────────────────────────────────────
# 词典加载（插件初始化时一次性全量加载）
# ─────────────────────────────────────────────

NODE_DIR = Path(__file__).parent
DICT_DIR = NODE_DIR / "dim_dictionaries"
USER_DICT_DIR = NODE_DIR / "output" / "step3_dictionaries"

DICTIONARIES: dict[str, dict] = {}  # dim -> dict 内容


def merge_dictionary_layers(base_dict: dict, user_dict: dict) -> dict:
    """Merge base and user dictionaries at runtime."""
    if not base_dict and not user_dict:
        return {"clusters": {}}

    merged = dict(base_dict or user_dict)
    merged_clusters = {}

    for key, cluster in (base_dict or {}).get("clusters", {}).items():
        merged_clusters[key] = dict(cluster)
        merged_clusters[key]["samples"] = list(cluster.get("samples", []))

    for key, cluster in (user_dict or {}).get("clusters", {}).items():
        if key in merged_clusters:
            base_samples = merged_clusters[key].get("samples", [])
            seen = set(base_samples)
            for sample in cluster.get("samples", []):
                if sample not in seen:
                    base_samples.append(sample)
                    seen.add(sample)
            merged_clusters[key]["samples"] = base_samples
        else:
            merged_clusters[key] = dict(cluster)
            merged_clusters[key]["samples"] = list(cluster.get("samples", []))

    merged["clusters"] = merged_clusters
    return merged


def load_all_dictionaries():
    """Load base dictionaries and merge user incremental dictionaries into memory."""
    DICTIONARIES.clear()
    for dim in DIMS:
        # gender 为硬编码维度，不依赖外部词典文件。
        if dim == "gender":
            DICTIONARIES[dim] = {"clusters": {}}
            continue

        base_path = DICT_DIR / f"{dim}_dict.json"
        user_path = USER_DICT_DIR / f"{dim}_dict.json"

        base_dict = {"clusters": {}}
        user_dict = {"clusters": {}}

        if base_path.exists():
            try:
                with open(base_path, encoding="utf-8") as f:
                    base_dict = json.load(f)
            except Exception as e:
                print(f"[PromptEngine] Warning: failed to load base dictionary {dim}_dict.json: {e}")
        else:
            print(f"[PromptEngine] Warning: base dictionary file not found: {base_path}")

        if user_path.exists():
            try:
                with open(user_path, encoding="utf-8") as f:
                    user_dict = json.load(f)
            except Exception as e:
                print(f"[PromptEngine] Warning: failed to load user dictionary {dim}_dict.json: {e}")

        DICTIONARIES[dim] = merge_dictionary_layers(base_dict, user_dict)


load_all_dictionaries()


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def get_style_list(dim: str) -> list[str]:
    """
    返回某个维度的 Style 下拉框选项列表。
    格式：[STYLE_RANDOM, STYLE_SKIP, cluster_key1, cluster_key2, ...]
    """
    clusters = DICTIONARIES.get(dim, {}).get("clusters", {})
    keys = list(clusters.keys())
    return [STYLE_RANDOM, STYLE_SKIP] + keys


def get_all_possible_style_values() -> list[str]:
    """
    返回所有维度的所有可能的 style 值（包括中英文显示名称）。
    用于 INPUT_TYPES 中的 style 参数定义，确保验证通过。
    """
    all_values = {STYLE_RANDOM, STYLE_SKIP}
    
    # 添加性别选项（硬编码，包括中英文）
    all_values.update(["man", "woman", "男性", "女性"])
    
    for dim in DIMS:
        clusters = DICTIONARIES.get(dim, {}).get("clusters", {})
        for key, cluster_data in clusters.items():
            # 添加内部 key
            all_values.add(key)
            # 添加英文显示名称
            canonical_name = cluster_data.get("canonical_name")
            if canonical_name:
                all_values.add(canonical_name)
            # 添加中文显示名称
            canonical_name_zh = cluster_data.get("canonical_name_zh")
            if canonical_name_zh:
                all_values.add(canonical_name_zh)
    
    return sorted(list(all_values))


def get_canonical_names(dim: str) -> list[str]:
    """
    返回某个维度的 canonical_name 列表（用于前端显示）。
    与 get_style_list 顺序一一对应（跳过前两个特殊选项）。
    """
    clusters = DICTIONARIES.get(dim, {}).get("clusters", {})
    return [v.get("canonical_name", k) for k, v in clusters.items()]


def is_known_dimension_value(dim: str, value: str) -> bool:
    """Return True if a text matches a known key/display phrase/sample for a dimension."""
    candidate = (value or "").strip()
    if not candidate:
        return False

    if dim == "gender":
        return candidate in {"man", "woman", "男性", "女性"}

    clusters = DICTIONARIES.get(dim, {}).get("clusters", {})
    for key, cluster in clusters.items():
        if candidate == key:
            return True
        if candidate == cluster.get("canonical_name", ""):
            return True
        if candidate == cluster.get("canonical_name_zh", ""):
            return True
        if candidate == cluster.get("canonical_phrase", ""):
            return True
        if candidate in cluster.get("samples", []):
            return True
    return False


def resolve_style_content(dim: str, style: str, variation: bool, rng: random.Random) -> str:
    """
    根据 style 选择和 variation 开关，返回最终的提示词内容。

    返回规则：
      - STYLE_SKIP   → ""
      - STYLE_RANDOM → 随机选一个 cluster，再按 variation 决定 phrase/sample
      - 具体 key     → 按 variation 决定 phrase/sample
    
    特殊处理：
      - gender 维度：直接返回 "man" 或 "woman"
    """
    # 特殊处理性别维度
    if dim == "gender":
        if style == STYLE_SKIP:
            return ""
        
        # 处理 STYLE_RANDOM（🎲 Random Style）
        if style == STYLE_RANDOM:
            # 直接从 GENDER_OPTIONS 中随机选择
            return rng.choice(GENDER_OPTIONS)
        
        # 处理中文显示名称到英文 key 的转换
        style_key = style
        if style == "男性":
            style_key = "man"
        elif style == "女性":
            style_key = "woman"
        
        if style_key not in GENDER_OPTIONS:
            return ""
        return style_key  # 返回 "man" 或 "woman"
    
    if style == STYLE_SKIP:
        return ""

    clusters = DICTIONARIES.get(dim, {}).get("clusters", {})
    if not clusters:
        return ""

    # 确定目标 cluster
    cluster = None
    cluster_key = None
    
    if style == STYLE_RANDOM:
        # 随机选择一个 cluster key
        cluster_key = rng.choice(list(clusters.keys()))
        cluster = clusters.get(cluster_key)
        if not cluster:
            print(f"[PromptEngine] Warning: random selection failed for dimension '{dim}'")
            return ""
    else:
        # 首先尝试直接用 style 作为 key 查找
        cluster = clusters.get(style)
        
        # 如果找不到，尝试通过显示名称查找（支持中英文）
        if not cluster:
            for key, cluster_data in clusters.items():
                # 检查是否匹配 canonical_name 或 canonical_name_zh
                if (cluster_data.get("canonical_name") == style or 
                    cluster_data.get("canonical_name_zh") == style):
                    cluster = cluster_data
                    cluster_key = key
                    break
        else:
            cluster_key = style
        
        # 如果还是找不到，返回空
        if not cluster:
            print(f"[PromptEngine] Warning: style '{style}' not found in dimension '{dim}'")
            return ""

    if variation:
        samples = cluster.get("samples", [])
        if samples:
            return rng.choice(samples)
        # fallback：samples 为空时退回 canonical_phrase
        return cluster.get("canonical_phrase", "")
    else:
        return cluster.get("canonical_phrase", "")


def join_parts(*parts: str) -> str:
    """
    将多个字符串拼接为提示词，自动跳过空字符串，逗号分隔。
    """
    return ", ".join(p.strip() for p in parts if p and p.strip())


def make_rng(seed: int, _unused: list) -> random.Random:
    """
    返回固定 seed 的 RNG 实例。
    ComfyUI 框架在每次执行前已根据 control_after_generate 模式
    更新好了传入的 seed 值，无需在此处重复处理。
    """
    return random.Random(seed)


# ─────────────────────────────────────────────
# PromptEngine Node — 单维度节点
# ─────────────────────────────────────────────

class PromptEngineNode:
    """
    单维度提示词节点。
    通过 prompt_in / prompt_out 串联多个节点，每个节点负责一个维度。
    """

    CATEGORY = "PromptEngine"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_out",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        # 默认用第一个维度的 style 列表；前端 JS 会动态替换
        default_dim = DIMS[0]
        # 使用包含所有维度所有可能值的列表，确保验证通过
        style_list = get_all_possible_style_values()
        
        # category 参数需要包含所有可能的显示名称（英文 key + 中文显示名）
        # 因为前端会修改 widget 的 options.values 为显示名称
        dim_values = list(DIMS)  # 内部 key
        for dim in DIMS:
            display = DIM_DISPLAY.get(dim, {})
            if display.get("en"):
                dim_values.append(display["en"])
            if display.get("zh"):
                dim_values.append(display["zh"])
        
        return {
            "required": {
                "category": (dim_values, {
                    "default": default_dim,
                    # 前端通过 dim_display 映射显示友好名称
                }),
                "style": (style_list, {
                    "default": STYLE_RANDOM,
                }),
                "variation": ("BOOLEAN", {
                    "default": False,
                    "label_on":  "Variation ON",
                    "label_off": "Variation OFF",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                }),
                "custom_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Override: type anything here to bypass Style selection",
                }),
            },
            "optional": {
                "prompt_in": ("STRING", {
                    "default": "",
                    "forceInput": True,
                }),
            },
        }

    def generate(
        self,
        category: str,
        style: str,
        variation: bool,
        seed: int,
        custom_text: str,
        prompt_in: str = "",
    ) -> tuple[str]:

        rng = make_rng(seed, [])
        
        # 将显示名称转换为内部 key（支持中英文显示名）
        dim_key = category
        if category in DIM_DISPLAY:
            dim_key = category  # 已经是 key
        else:
            # 尝试通过显示名称查找 key
            for key, display in DIM_DISPLAY.items():
                if display.get("en") == category or display.get("zh") == category:
                    dim_key = key
                    break
        
        # 如果找不到对应的 key，直接使用原值
        if dim_key not in DIMS:
            dim_key = category

        # 优先级：custom_text > style 选择
        if custom_text and custom_text.strip():
            dim_content = custom_text.strip()
        else:
            dim_content = resolve_style_content(dim_key, style, variation, rng)
        
        # 特殊处理：如果当前维度是 gender，且 prompt_in 的最后部分是 ethnicity
        if dim_key == "gender" and dim_content and prompt_in:
            # 检查前一个输出是否是 ethnicity（简单的判断：prompt_in 只有 1-2 个单词且没有逗号）
            parts = prompt_in.split(", ")
            last_part = parts[-1].strip()
            # 如果最后部分看起来像种族描述（短词，且不是完整的句子）
            if last_part and len(last_part.split()) <= 2 and ',' not in last_part:
                # 将 gender 与 ethnicity 拼接
                prefix = ", ".join(parts[:-1]) if len(parts) > 1 else ""
                combined = f"{last_part} {dim_content}"  # 如 "White woman"
                prompt_out = join_parts(prefix, combined)
                return (prompt_out,)

        # 特殊处理：如果当前维度是 hair_style，且 prompt_in 的最后部分是 hair_color
        if dim_key == "hair_style" and dim_content and prompt_in:
            parts = prompt_in.split(", ")
            last_part = parts[-1].strip()
            if is_known_dimension_value("hair_color", last_part):
                prefix = ", ".join(parts[:-1]) if len(parts) > 1 else ""
                combined = f"{last_part} {dim_content}"
                prompt_out = join_parts(prefix, combined)
                return (prompt_out,)
        
        prompt_out = join_parts(prompt_in, dim_content)
        return (prompt_out,)


# ─────────────────────────────────────────────
# PromptEngine Full — 全维度节点
# ─────────────────────────────────────────────

def _build_full_input_types():
    """动态构建 PromptEngineFull 的 INPUT_TYPES，避免重复代码。"""
    required = {}

    # 每个维度一个 style 下拉框
    for dim in DIMS:
        # 使用包含所有维度所有可能值的列表，确保验证通过
        style_list = get_all_possible_style_values()
        required[f"{dim}_style"] = (style_list, {
            "default": STYLE_RANDOM,
        })

    # 全局控制
    required["variation"] = ("BOOLEAN", {
        "default": False,
        "label_on":  "Variation ON",
        "label_off": "Variation OFF",
    })
    required["seed"] = ("INT", {
        "default": 0,
        "min": 0,
        "max": 2**31 - 1,
    })
    # control_after_generate 由 ComfyUI 框架自动附加到 seed widget，无需声明

    return {
        "required": required,
        "optional": {
            "custom_text": ("STRING", {
                "default": "",
                "multiline": False,
                "forceInput": True,
                "placeholder": "Optional prefix (e.g. LoRA trigger words)",
            }),
        },
    }


class PromptEngineFull:
    """
    全维度提示词节点。
    21 个维度全部展开，适合快速生成和批量场景。
    """

    CATEGORY = "PromptEngine"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_out",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return _build_full_input_types()

    def generate(self, variation: bool, seed: int,
                 custom_text: str = "", **kwargs) -> tuple[str]:

        rng = make_rng(seed, [])

        parts = []

        # 前缀
        if custom_text and custom_text.strip():
            parts.append(custom_text.strip())

        # 按标准顺序拼接各维度
        for dim in DIMS:
            style = kwargs.get(f"{dim}_style", STYLE_SKIP)
            content = resolve_style_content(dim, style, variation, rng)
            if content:
                # 特殊处理：ethnicity 和 gender 的拼接
                if dim == "ethnicity":
                    # 检查是否有 gender，如果有则拼接
                    gender_style = kwargs.get("gender_style", STYLE_SKIP)
                    gender_content = resolve_style_content("gender", gender_style, variation, rng)
                    if gender_content:
                        combined = f"{content} {gender_content}"  # 如 "a Chinese woman"
                        parts.append(combined)
                    else:
                        parts.append(content)
                elif dim == "gender":
                    # gender 已经和 ethnicity 拼接了，这里跳过
                    continue
                elif dim == "hair_style":
                    hair_color_style = kwargs.get("hair_color_style", STYLE_SKIP)
                    hair_color_content = resolve_style_content("hair_color", hair_color_style, variation, rng)
                    if hair_color_content:
                        parts.append(f"{hair_color_content} {content}")
                    else:
                        parts.append(content)
                elif dim == "hair_color":
                    # hair_color 已经和 hair_style 拼接了，这里跳过独立输出
                    hair_style_style = kwargs.get("hair_style_style", STYLE_SKIP)
                    hair_style_content = resolve_style_content("hair_style", hair_style_style, variation, rng)
                    if hair_style_content:
                        continue
                    parts.append(content)
                else:
                    parts.append(content)

        prompt_out = ", ".join(parts)
        return (prompt_out,)


# ─────────────────────────────────────────────
# API 路由：向前端暴露词典数据
# ─────────────────────────────────────────────

def register_api_routes():
    """注册 API 路由，供前端 JS 获取词典数据。"""
    try:
        from aiohttp import web
        from server import PromptServer

        routes = PromptServer.instance.routes

        @routes.get("/promptengine/dictionaries")
        async def get_dictionaries(request):
            """
            返回所有维度的词典摘要（canonical_name、canonical_name_zh、tags）。
            前端用于构建动态下拉框。
            """
            load_all_dictionaries()
            result = {}
            for dim in DIMS:
                # 特殊处理 gender 维度（硬编码，无需词典）
                if dim == "gender":
                    result[dim] = {
                        "display": DIM_DISPLAY.get(dim, {"en": dim, "zh": dim}),
                        "clusters": {
                            "man": {
                                "canonical_name": "man",
                                "canonical_name_zh": "男性",
                                "tags": []
                            },
                            "woman": {
                                "canonical_name": "woman",
                                "canonical_name_zh": "女性",
                                "tags": []
                            }
                        }
                    }
                    continue
                
                d = DICTIONARIES.get(dim, {})
                clusters_summary = {}
                for key, cluster in d.get("clusters", {}).items():
                    clusters_summary[key] = {
                        "canonical_name":    cluster.get("canonical_name", key),
                        "canonical_name_zh": cluster.get("canonical_name_zh", ""),
                        "tags":              cluster.get("tags", []),
                    }
                result[dim] = {
                    "display": DIM_DISPLAY.get(dim, {"en": dim, "zh": dim}),
                    "clusters": clusters_summary,
                }
            return web.json_response(result)

        print("[PromptEngine] API route registered: GET /promptengine/dictionaries")

    except Exception as e:
        print(f"[PromptEngine] API route registration failed (not running inside ComfyUI?): {e}")


# ─────────────────────────────────────────────
# LLM 配置节点
# ─────────────────────────────────────────────

class LLMConfigNode:
    """
    LLM 配置节点
    为 Step 1 和 Step 3 提供统一的 LLM 配置
    支持任何符合 OpenAI API 规范的服务
    """
    
    CATEGORY = "PromptEngine/Config"
    FUNCTION = "get_config"
    RETURN_TYPES = ("LLM_CONFIG",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base": ("STRING", {
                    "default": "http://localhost:8080/v1",
                    "multiline": False,
                    "placeholder": "API Base URL (如：http://localhost:8080/v1)"
                }),
                "api_key": ("STRING", {
                    "default": "not-needed",
                    "multiline": False,
                    "placeholder": "API Key (本地模型填 not-needed)"
                }),
                "model_name": ("STRING", {
                    "default": "local-model",
                    "multiline": False,
                    "placeholder": "模型名称 (如：Qwen3.5-35B, gpt-4)"
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "label": "Temperature"
                }),
                "max_tokens": ("INT", {
                    "default": 1200,
                    "min": 100,
                    "max": 8000,
                    "step": 100,
                    "label": "Max Tokens"
                }),
                "concurrency": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "display": "slider",
                    "label": "Concurrency"
                }),
            }
        }
    
    def get_config(self, api_base, api_key, model_name, 
                   temperature=0.1, max_tokens=1200, concurrency=4):
        """
        生成 LLM 配置字典
        
        Returns:
            tuple: (config_dict,)
        """
        config = {
            "base_url": api_base.strip(),
            "api_key": api_key.strip() if api_key.strip() else "not-needed",
            "model": model_name.strip(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
        }
        
        print("\n[LLM Config] Configuration loaded:")
        print(f"  API Base: {config['base_url']}")
        print(f"  Model: {config['model']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  Max Tokens: {config['max_tokens']}")
        print(f"  Concurrency: {config['concurrency']}")
        
        return (config,)


# ─────────────────────────────────────────────
# Step 1-3 工具节点
# ─────────────────────────────────────────────

class Step1DimensionExtract:
    """
    Step 1: 维度提取节点
    从文本文件读取原始提示词，调用 LLM 提取 21 个维度，输出结构化 JSON
    """
    
    CATEGORY = "PromptEngine/Tools"
    FUNCTION = "extract_dimensions"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "json_path")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {
                    "default": "prompts.txt",
                    "multiline": False,
                    "placeholder": "输入文件路径（相对于工作目录）"
                }),
            },
            "optional": {
                "llm_config": ("LLM_CONFIG",),  # ← 新增 LLM 配置输入
                "test_mode": ("BOOLEAN", {
                    "default": False,
                    "label": "Test Mode (仅前 10 条)"
                }),
            }
        }
    
    def extract_dimensions(self, input_file, llm_config=None, test_mode=False):
        import asyncio
        from datetime import datetime
        from .tools.step1_dimension_extract import run_extraction
        
        try:
            print("\n[Step 1] Starting dimension extraction...")
            print(f"Input file: {input_file}")
            if llm_config:
                print(f"LLM config: {llm_config.get('model', 'unknown')} @ {llm_config.get('base_url', 'unknown')}")
            print("Output subdirectory: step1_json")
            print(f"Test mode: {'on' if test_mode else 'off'}")
            
            # Step 1 输出固定写入插件目录下的 output/step1_json
            plugin_dir = Path(__file__).parent
            output_dir = plugin_dir / "output" / "step1_json"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成带时间戳的输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(input_file).stem
            output_filename = f"{input_name}_{timestamp}.json"
            output_file = output_dir / output_filename
            
            # 调用 Step 1，传入配置
            result_file = run_extraction(
                input_path=input_file,
                output_path=str(output_file),  # ← 明确指定输出路径
                llm_config=llm_config,  # ← 传入 LLM 配置
                test_mode=test_mode
            )
            
            print(f"[Step 1] Completed. Output: {result_file}")
            
            # 统计结果
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total = len(data)
                errors = sum(1 for v in data.values() if '_error' in v)
                
            msg = (
                "Step 1 completed\n"
                f"Total records: {total}\n"
                f"Successful: {total - errors}\n"
                f"Errors: {errors}\n"
                f"Saved file: {result_file}"
            )
            print(msg)
            
            return (msg, result_file)
            
        except Exception as e:
            error_msg = f"Step 1 failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg, "")


class Step2Clustering:
    """
    Step 2: 聚类分析节点
    读取 Step 1 的 JSON 输出，对每个维度进行语义聚类
    """
    
    CATEGORY = "PromptEngine/Tools"
    FUNCTION = "run_clustering"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("summary", "cluster_dir")
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_json": ("STRING", {
                    "default": "data/02_merged_normal_20260211_001-026.json",
                    "multiline": False,
                    "placeholder": "Step 1 输出的 JSON 文件路径"
                }),
            },
            "optional": {
                "enable_noise_recluster": ("BOOLEAN", {
                    "default": True,
                    "label": "启用噪声二次聚类"
                }),
                "primary_min_cluster_size": ("INT", {
                    "default": 6,
                    "min": 2,
                    "max": 20,
                    "step": 1
                }),
                "primary_min_samples": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                "noise_min_cluster_size": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 10,
                    "step": 1
                }),
                "noise_min_samples": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }
    
    def run_clustering(self, input_json, enable_noise_recluster=True, 
                      primary_min_cluster_size=6, primary_min_samples=3,
                      noise_min_cluster_size=4, noise_min_samples=2):
        from .tools.step2_clustering import run_clustering_pipeline
        
        try:
            print("\n[Step 2] Starting clustering...")
            print(f"Input: {input_json}")
            print(f"Noise reclustering: {'enabled' if enable_noise_recluster else 'disabled'}")
            
            # 调用 Step 2 包装函数
            results, summary_markdown, cluster_dir = run_clustering_pipeline(
                input_path=input_json,
                enable_noise_recluster=enable_noise_recluster,
                primary_min_cluster_size=primary_min_cluster_size,
                primary_min_samples=primary_min_samples,
                noise_min_cluster_size=noise_min_cluster_size,
                noise_min_samples=noise_min_samples
            )
            
            print("[Step 2] Completed.")
            
            # 汇总统计
            total_clusters = sum(r['main_clusters'] + r.get('sub_clusters', 0) 
                               for r in results.values())

            msg = summary_markdown
            print(msg)
            
            return (msg, cluster_dir)
            
        except Exception as e:
            error_msg = f"Step 2 failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg, "")


class Step3DictionaryGen:
    """
    Step 3: 词典生成节点
    基于 Step 2 的聚类结果，生成/更新维度词典
    """
    
    CATEGORY = "PromptEngine/Tools"
    FUNCTION = "generate_dictionary"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cluster_input_dir": ("STRING", {
                    "default": str((Path(__file__).parent / "output" / "step2_clusters").resolve()),
                    "multiline": False,
                    "placeholder": "Step 2 聚类结果目录"
                }),
                "llm_config": ("LLM_CONFIG",),
            }
        }
    
    def generate_dictionary(self, cluster_input_dir=None, llm_config=None):
        from .tools.step3_dictionary_gen import run_dictionary_generation
        
        try:
            if cluster_input_dir is None:
                cluster_input_dir = str((Path(__file__).parent / "output" / "step2_clusters").resolve())
            base_dict_dir = str((Path(__file__).parent / "dim_dictionaries").resolve())

            print("\n[Step 3] Starting dictionary generation...")
            print(f"Base dictionary dir: {base_dict_dir}")
            print(f"Cluster input dir: {cluster_input_dir}")
            if llm_config:
                print(f"LLM config: {llm_config.get('model', 'unknown')} @ {llm_config.get('base_url', 'unknown')}")
            
            # 调用 Step 3 包装函数
            results = run_dictionary_generation(
                base_dict_dir=base_dict_dir,
                input_dir=cluster_input_dir,
                llm_config=llm_config,
            )

            load_all_dictionaries()
            
            print("[Step 3] Completed.")
            
            output_dir = Path(__file__).parent / "output" / "step3_dictionaries"
            msg = (
                "Step 3 completed\n"
                f"Dimensions updated: {len(results)}\n"
                f"Output directory: {output_dir.resolve()}"
            )
            print(msg)
            
            return (msg,)
            
        except Exception as e:
            error_msg = f"Step 3 failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg,)


register_api_routes()


# ─────────────────────────────────────────────
# 节点注册
# ─────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    # PromptEngine 核心节点
    "PromptEngineNode": PromptEngineNode,
    "PromptEngineFull": PromptEngineFull,
    
    # LLM 配置节点
    "LLMConfigNode": LLMConfigNode,
    
    # Step 1-3 工具节点
    "Step1DimensionExtract": Step1DimensionExtract,
    "Step2Clustering": Step2Clustering,
    "Step3DictionaryGen": Step3DictionaryGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # PromptEngine 核心节点
    "PromptEngineNode": "PromptEngine Node",
    "PromptEngineFull": "PromptEngine Full (All Dimensions)",
    
    # LLM 配置节点
    "LLMConfigNode": "LLM Config (Step 1 & 3)",
    
    # Step 1-3 工具节点
    "Step1DimensionExtract": "Step 1: Dimension Extract",
    "Step2Clustering": "Step 2: Clustering Analysis",
    "Step3DictionaryGen": "Step 3: Dictionary Generator",
}
