"""
Offline regression checks for core PromptEngine behavior.

Runs without a ComfyUI runtime by mocking the `server` module and loading
`nodes.py` directly from this plugin directory.
"""

import json
import random
import sys
import types
from pathlib import Path


PLUGIN_DIR = Path(__file__).parent
MOCK_DICT_DIR = PLUGIN_DIR / "output" / "test_mock_dicts"
MOCK_DICT_DIR.mkdir(parents=True, exist_ok=True)

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

MOCK_CLUSTER = {
    "black_lace_lingerie": {
        "canonical_name": "Black Lace Lingerie",
        "canonical_name_zh": "黑色蕾丝内衣",
        "canonical_phrase": "black lace lingerie",
        "samples": [
            "black lace-trimmed lingerie",
            "black lace lingerie set with thin straps",
            "black lace lingerie with bow detail",
        ],
        "tags": ["lace", "black", "lingerie", "sheer", "delicate"],
    },
    "white_shirt_jeans": {
        "canonical_name": "White Shirt and Jeans",
        "canonical_name_zh": "白衬衫牛仔裤",
        "canonical_phrase": "white button-up shirt, fitted blue jeans",
        "samples": [
            "white shirt, blue jeans",
            "white cotton shirt, ripped jeans",
        ],
        "tags": ["casual", "white", "denim", "shirt", "everyday"],
    },
}


def build_mock_dicts():
    for dim in DIMS:
        payload = {
            "dimension": dim,
            "version": "1.0",
            "generated_date": "2026-03-19",
            "clusters": MOCK_CLUSTER if dim == "outfit" else {
                "mock_cluster_a": {
                    "canonical_name": f"{dim.title()} Option A",
                    "canonical_name_zh": f"{dim} 选项A",
                    "canonical_phrase": f"mock {dim} phrase A",
                    "samples": [f"sample {dim} 1", f"sample {dim} 2"],
                    "tags": ["mock", "test"],
                }
            },
        }
        with open(MOCK_DICT_DIR / f"{dim}_dict.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def load_nodes_module():
    fake_server = types.ModuleType("server")

    class FakePromptServer:
        instance = None

        class routes:
            @staticmethod
            def get(_path):
                return lambda func: func

    fake_server.PromptServer = FakePromptServer
    sys.modules["server"] = fake_server

    sys.path.insert(0, str(PLUGIN_DIR))
    import nodes as module  # noqa: E402

    module.DICT_DIR = MOCK_DICT_DIR
    module.DICTIONARIES.clear()
    module.load_all_dictionaries()
    return module


def assert_equal(actual, expected, label):
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"
    print(f"[PASS] {label}")


def main():
    build_mock_dicts()
    module = load_nodes_module()

    style_list = module.get_style_list("outfit")
    assert style_list[0] == module.STYLE_RANDOM
    assert style_list[1] == module.STYLE_SKIP
    assert "black_lace_lingerie" in style_list
    print("[PASS] get_style_list")

    rng = random.Random(42)
    assert_equal(
        module.resolve_style_content("outfit", module.STYLE_SKIP, False, rng),
        "",
        "resolve_style_content skip",
    )
    assert_equal(
        module.resolve_style_content("outfit", "black_lace_lingerie", False, rng),
        "black lace lingerie",
        "resolve_style_content canonical phrase",
    )

    varied = module.resolve_style_content("outfit", "black_lace_lingerie", True, rng)
    assert varied in MOCK_CLUSTER["black_lace_lingerie"]["samples"]
    print("[PASS] resolve_style_content variation")

    assert_equal(module.join_parts("a", "", "b"), "a, b", "join_parts")

    node = module.PromptEngineNode()
    out, = node.generate(
        category="outfit",
        style="black_lace_lingerie",
        variation=False,
        seed=0,
        custom_text="",
        prompt_in="beautiful woman",
    )
    assert_equal(out, "beautiful woman, black lace lingerie", "PromptEngineNode.generate")

    full_node = module.PromptEngineFull()
    kwargs = {f"{dim}_style": module.STYLE_SKIP for dim in module.DIMS}
    kwargs["outfit_style"] = "black_lace_lingerie"
    kwargs["lighting_style"] = "mock_cluster_a"
    out, = full_node.generate(
        variation=False,
        seed=0,
        custom_text="1girl",
        **kwargs,
    )
    assert_equal(
        out,
        "1girl, black lace lingerie, mock lighting phrase A",
        "PromptEngineFull.generate",
    )

    color_key = next(iter(module.DICTIONARIES["hair_color"]["clusters"].keys()))
    style_key = next(iter(module.DICTIONARIES["hair_style"]["clusters"].keys()))

    out, = node.generate(
        category="hair_style",
        style=style_key,
        variation=False,
        seed=0,
        custom_text="",
        prompt_in="mock hair_color phrase A",
    )
    assert_equal(
        out,
        "mock hair_color phrase A, mock hair_style phrase A".replace(", ", " ", 1),
        "PromptEngineNode hair_color + hair_style merge",
    )

    kwargs = {f"{dim}_style": module.STYLE_SKIP for dim in module.DIMS}
    kwargs["hair_color_style"] = color_key
    kwargs["hair_style_style"] = style_key
    out, = full_node.generate(
        variation=False,
        seed=0,
        custom_text="",
        **kwargs,
    )
    assert_equal(
        out,
        "mock hair_color phrase A mock hair_style phrase A",
        "PromptEngineFull hair_color + hair_style merge",
    )

    print("\nAll offline checks passed.")


if __name__ == "__main__":
    main()
