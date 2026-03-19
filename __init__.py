"""
ComfyUI-PromptEngine
提示词引擎插件 — 基于 21 维度词典，通过可视化界面组合 T2I 提示词

节点：
  - PromptEngine Node  : 单维度节点，多节点串联
  - PromptEngine Full  : 全维度节点，21 个维度全部展开
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
