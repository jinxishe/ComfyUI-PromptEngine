# ComfyUI-PromptEngine v1.2.1

ComfyUI-PromptEngine is now ready for its first public GitHub release.

This plugin provides a dictionary-based prompt composition workflow for ComfyUI, with 21 visual dimensions covering appearance, outfit, pose, scene, camera, lighting, color, and style. It includes both modular single-dimension nodes and an all-in-one full editor node, plus a Step 1-3 toolchain for extracting dimensions from prompt datasets, clustering related phrases, and generating incremental user dictionaries.

## Included in this release

- `PromptEngineNode` for chainable dimension-by-dimension prompt building
- `PromptEngineFull` for full prompt editing in a single node
- bundled base dictionaries across 21 dimensions
- `LLMConfigNode`, `Step1DimensionExtract`, `Step2Clustering`, and `Step3DictionaryGen`
- public English and Chinese documentation
- GitHub and ComfyUI Registry publication metadata

## Improvements in v1.2.1

- fixed Step 1 output handling and removed `output_subdir`
- fixed Step 1 prompt filtering for overlong prompts
- fixed Step 1 and Step 3 API key passthrough
- fixed Step 2 standalone entrypoint unpacking
- refreshed tests and public-facing documentation

## Notes

- Step 1 input must be a plain text file with one complete prompt per line.
- Best results come from source prompts that cover most of the 21 target dimensions.
- Step 2 may download the embedding model on first run if it is not available locally.

Repository:
https://github.com/jinxishe/ComfyUI-PromptEngine
