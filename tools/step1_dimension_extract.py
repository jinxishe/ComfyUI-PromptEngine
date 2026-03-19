"""
Step 1: 批量将原始提示词解析为20维度结构化JSON

依赖:
    pip install openai tqdm

前提:
    llama.cpp 服务器已启动，例如：
    ./llama-server -m Qwen3.5-35B-A3B-heretic-v2-mxfp4_moe.gguf \
        --host 0.0.0.0 --port 8080 \
        -c 2048 -np 2 --no-mmap

用法:
    python step1_dimension_extract.py                      # 处理 prompts.txt
    python step1_dimension_extract.py my_prompts.txt       # 指定输入文件
    python step1_dimension_extract.py --test               # 只跑前10条验证效果
    python step1_dimension_extract.py --concurrency 8      # 覆盖并发数
"""

import json
import sys
import asyncio
import argparse
import threading
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# ─────────────────────────────────────────────
# 配置（内部配置，LLM 配置通过 LLMConfigNode 提供）
# ─────────────────────────────────────────────

CONFIG = {
    "output_file":         "",            # 由节点动态设置
    "checkpoint_interval": 20,            # 每处理 N 条保存一次断点
}

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise T2I prompt analyzer for portrait photography.
Extract 20 dimensions from the user's prompt.

Rules:
- Use English phrases that preserve all meaningful details
- Use empty string "" if a dimension is not mentioned
- subject_appearance: body type + skin tone only, NOT hair
                 e.g. "slender, fair skin", "athletic build, tanned"
- outfit        : ALL clothing and garments on or held by the subject,
                  including tops, bottoms, dresses, outerwear, jackets,
                  cardigans, robes, and any garment held in hand
                  (treated as a special wearing style, e.g. "denim jacket
                  held in hand"). No length limit — preserve all layering
                  details, materials, patterns, cuts, and notable features.
                  e.g. "pale yellow halter bikini top with deep plunging
                       neckline, fitted denim jeans with visible button
                       and seams"
                  e.g. "purple floral patterned bandeau top, sheer white
                       lace cropped jacket draped over shoulders, matching
                       purple floral patterned skirt with high slit"
- accessories   : jewelry, bags, hats, footwear, belts, sunglasses,
                  hair accessories ONLY. Never include garments here.
                  e.g. "brown boots, silver necklace, black belt"
- pose          : combine body orientation + arms + legs in one phrase
                  e.g. "lying supine, arms along body, legs slightly apart"
- body_direction: relation to camera
                  e.g. "facing camera", "three-quarter turn left", "back to camera"
- composition   : framing, crop, depth of field, foreground elements,
                  and spatial layout — preserve all details
                  e.g. "medium shot centered, shallow depth of field,
                       faucets in foreground softly out of focus"
- lighting      : combine type + quality in one phrase
                  e.g. "soft natural daylight from above", "dramatic studio rim light"
- color_grade   : overall color tone
                  e.g. "warm earthy tones", "cool desaturated", "high contrast vivid"
- visual_style  : overall aesthetic
                  e.g. "editorial fashion", "natural candid", "glamour photography"
- shot_distance : one of: "extreme close-up" / "close-up" / "medium" / "full body" / "wide shot"
- background_props: list all visible background elements, props, and
                  environmental details as completely as possible
                  e.g. "white ceramic sink, vintage brass faucets,
                       wooden paneling, traditional Japanese shoji window"

Return exactly this JSON and nothing else:
{
  "ethnicity": "",
  "age_appearance": "",
  "subject_appearance": "",
  "hair_style": "",
  "hair_color": "",
  "outfit": "",
  "accessories": "",
  "pose": "",
  "body_direction": "",
  "expression": "",
  "gaze": "",
  "location_type": "",
  "background_props": "",
  "atmosphere": "",
  "shot_angle": "",
  "shot_distance": "",
  "composition": "",
  "lighting": "",
  "color_grade": "",
  "visual_style": ""
}\
"""

USER_TEMPLATE = "{prompt}"

DIMS = [
    "ethnicity", "age_appearance", "subject_appearance",
    "hair_style", "hair_color",
    "outfit", "accessories",
    "pose", "body_direction",
    "expression", "gaze",
    "location_type", "background_props", "atmosphere",
    "shot_angle", "shot_distance", "composition",
    "lighting", "color_grade", "visual_style",
]

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # 过滤异常长的条目（正常提示词不超过3000字符）
    MAX_CHARS = 3000
    filtered = []
    skipped = []
    for i, line in enumerate(lines):
        if len(line) <= MAX_CHARS:
            filtered.append(line)
        else:
            skipped.append(i)

    if skipped:
        print(f"跳过 {len(skipped)} 条异常长提示词（行号: {skipped}）")

    print(f"Loaded {len(filtered)} prompts from {path}")
    return filtered


def parse_response(text: str) -> dict:
    """从模型输出中提取 JSON，容忍常见格式问题"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    # 截取第一个 { 到最后一个 }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e+1]
    return json.loads(text)


def save(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Checkpoint found. {len(data)} items already completed; resuming...")
        return data
    return {}


def print_stats(results: dict):
    total  = len(results)
    errors = sum(1 for v in results.values() if "_error" in v)
    valid  = total - errors
    valid_records = [v for v in results.values() if "_error" not in v]

    print(f"\n{'─'*50}")
    print(f"  Total {total}    Success {valid}    Failed {errors}")
    print(f"{'─'*50}")

    if not valid_records:
        return

    print("  Dimension fill rates:")
    for dim in DIMS:
        filled = sum(1 for r in valid_records if r.get(dim, "").strip())
        rate   = filled / valid * 100
        bar    = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        print(f"  {dim:<22} {bar} {rate:5.1f}%")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
# 核心：异步并发提取
# ─────────────────────────────────────────────

async def run(prompts: list[str], cfg: dict) -> dict:
    client    = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg.get("api_key", "not-needed"))
    semaphore = asyncio.Semaphore(cfg["concurrency"])
    out_path  = cfg["output_file"]

    results = load_checkpoint(out_path)
    done    = set(results.keys())
    pending = [(i, p) for i, p in enumerate(prompts)
               if f"p_{i:07d}" not in done]

    if not pending:
        print("All items have already been processed.")
        return results

    print(f"{len(pending)} items pending ({len(done)} already completed and skipped)\n")

    async def extract_one(idx: int, prompt: str) -> tuple[str, dict]:
        cid = f"p_{idx:07d}"
        async with semaphore:
            raw = ""
            try:
                resp = await client.chat.completions.create(
                    model       = cfg["model"],
                    max_tokens  = cfg["max_tokens"],
                    temperature = cfg["temperature"],
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": USER_TEMPLATE.format(prompt=prompt)},
                    ]
                )
                raw = resp.choices[0].message.content
                parsed = parse_response(raw)
                parsed["_source"] = prompt
                return cid, parsed

            except json.JSONDecodeError as e:
                return cid, {"_error": f"json_parse: {e}", "_raw": raw[:120]}

            except Exception as e:
                return cid, {"_error": str(e)}

    tasks     = [extract_one(i, p) for i, p in pending]
    completed = 0

    async for fut in atqdm(asyncio.as_completed(tasks),
                           total=len(tasks), desc="提取"):
        cid, result = await fut
        results[cid] = result
        completed += 1
        if completed % cfg["checkpoint_interval"] == 0:
            save(results, out_path)

    save(results, out_path)
    return results


def run_async_compatible(coro):
    """
    在普通脚本环境下直接运行协程；
    如果当前线程已经有 event loop（如 ComfyUI），则切到独立线程执行。
    """
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
# 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="T2I提示词结构化抽取 Step 1")
    parser.add_argument("input",   nargs="?", default="prompts.txt",
                        help="Input file, one prompt per line (default: prompts.txt)")
    parser.add_argument("--test",  action="store_true",
                        help="Process only the first 10 prompts for testing")
    parser.add_argument("--concurrency", type=int,
                        help="Override the configured concurrency")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.concurrency:
        cfg["concurrency"] = args.concurrency

    # 仅在外部未指定输出路径时，才使用默认输出文件名
    input_path = Path(args.input)
    if not cfg.get("output_file"):
        cfg["output_file"] = str(input_path.with_suffix(".json"))

    prompts = load_prompts(args.input)

    if args.test:
        prompts = prompts[:10]
        if cfg["output_file"] == str(input_path.with_suffix(".json")):
            cfg["output_file"] = "extracted_test.json"
        print(f"[Test Mode] Processing only the first 10 items. Output: {cfg['output_file']}\n")

    print(f"Server: {cfg['base_url']}")
    print(f"Concurrency: {cfg['concurrency']}")
    print(f"Output: {cfg['output_file']}\n")

    results = run_async_compatible(run(prompts, cfg))
    print_stats(results)

    # 测试模式额外打印前3条结果供检查
    if args.test:
        print("Preview of the first 3 results:")
        for k, v in sorted(results.items())[:3]:
            print(f"\n[{k}]")
            if "_error" in v:
                print(f"  ERROR: {v}")
            else:
                for dim in DIMS:
                    val = v.get(dim, "")
                    if val:
                        print(f"  {dim:<22} {val}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
# ComfyUI 节点调用接口
# ─────────────────────────────────────────────

def run_extraction(input_path="prompts.txt", output_path=None, llm_config=None, concurrency=4, test_mode=False):
    """
    包装函数，供 ComfyUI 节点调用
    
    Args:
        input_path: Input text file path
        output_path: Output JSON file path (optional, defaults to input filename)
        llm_config: LLM config dict (optional, uses built-in defaults if omitted)
                    {
                        "base_url": "http://localhost:8080/v1",
                        "api_key": "not-needed",
                        "model": "local-model",
                        "temperature": 0.1,
                        "max_tokens": 1200,
                        "concurrency": 4,
                    }
        concurrency: Concurrency limit
        test_mode: Whether to test only the first 10 items
    
    Returns:
        Output JSON file path
    """
    import sys
    from io import StringIO
    
    # 声明全局变量（必须在最开头）
    global CONFIG
    
    # 构建配置：优先使用传入的 llm_config
    # 如果没有 llm_config，使用默认值（保证向后兼容）
    base_config = CONFIG.copy()
    
    if llm_config:
        # 完全使用 llm_config（用户提供）
        base_config.update(llm_config)
    else:
        # 降级使用默认配置（向后兼容，不连接 Config节点时使用默认值）
        base_config.update({
            "base_url": "http://localhost:8080/v1",
            "api_key": "not-needed",
            "model": "local-model",
            "temperature": 0.1,
            "max_tokens": 1200,
            "concurrency": concurrency,  # 使用参数中的 concurrency
        })
    
    # 临时修改 sys.argv 以兼容现有逻辑
    old_argv = sys.argv
    old_stdout = sys.stdout
    
    try:
        # 重定向 stdout 以捕获 print 输出
        sys.stdout = mystdout = StringIO()
        
        # 构建参数
        args_list = [input_path]
        if test_mode:
            args_list.append("--test")
        if concurrency != 4 and "--concurrency" not in args_list:
            args_list.extend(["--concurrency", str(concurrency)])
        
        sys.argv = ["step1_dimension_extract"] + args_list
        
        # 设置输出路径
        if output_path:
            base_config["output_file"] = output_path
        else:
            input_p = Path(input_path)
            base_config["output_file"] = str(input_p.with_suffix(".json"))
        
        # 执行主函数（临时修改 CONFIG）
        old_config = CONFIG.copy()
        CONFIG = base_config
        
        main()
        
        # 恢复配置
        CONFIG = old_config
        
        # 打印日志
        sys.stdout = old_stdout
        print(mystdout.getvalue())
        
        # 返回输出文件路径
        return base_config["output_file"]
        
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
