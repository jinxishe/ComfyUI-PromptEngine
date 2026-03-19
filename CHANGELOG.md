# Changelog

## v1.2.1 - 2026-03-19

- removed `output_subdir` from `Step1DimensionExtract`; Step 1 output is fixed to `output/step1_json/`
- fixed Step 1 prompt filtering so overlong prompts are actually excluded
- fixed Step 1 and Step 3 to honor the configured `api_key`
- fixed `tools/step2_clustering.py` standalone entrypoint unpacking
- replaced the stale offline test script with a smaller working regression test
- refreshed retained documentation to match the current plugin layout and behavior

## v1.2.0 - 2026-03-19

- added hardcoded `gender` dimension
- added automatic `ethnicity + gender` phrase merging
- stabilized bilingual frontend display logic
- renamed plugin folder to `ComfyUI-PromptEngine`

## v1.0.0 - 2026-03-18

- initial ComfyUI plugin release
- added `PromptEngineNode` and `PromptEngineFull`
- added bundled 21-dimension dictionary workflow
