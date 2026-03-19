# Publishing Guide

This file is for maintainers preparing the public GitHub release and ComfyUI Registry publication.

## Required metadata

Before the first registry publish, update these fields in `pyproject.toml`:

- `project.name`
  Must be globally unique in the ComfyUI Registry.
  It is immutable after creation.
  Registry docs recommend not including `ComfyUI` in this field.
- `project.urls.Repository`
- `project.urls.Documentation`
- `project.urls.Bug Tracker`
- `tool.comfy.PublisherId`

Optional fields you can add later:

- `tool.comfy.Icon`
- `tool.comfy.Banner`

## GitHub release checklist

1. Create the public GitHub repository.
2. Push this plugin as the repository root.
3. Update `pyproject.toml` placeholders with the real repository URL and final node id.
4. Confirm `LICENSE` matches the license you want to publish under.
5. Review `.gitignore` and ensure runtime outputs are not tracked.
6. Create repository secret `REGISTRY_ACCESS_TOKEN` with your Comfy Registry publishing API key.

## Registry publishing steps

1. Create a publisher account on Comfy Registry.
2. Create a publishing API key for that publisher.
3. Fill in `tool.comfy.PublisherId` in `pyproject.toml`.
4. Bump `project.version` for each release using semantic versioning.
5. Publish with one of the following methods.

### Option A: Comfy CLI

```bash
comfy node publish
```

### Option B: GitHub Actions

The included workflow at `.github/workflows/publish_action.yml` publishes automatically when `pyproject.toml` changes on `main`.

## Notes

- Once a version is published to the Registry, it cannot be modified in place.
- Registry metadata requirements and publishing flow were checked against the ComfyUI docs on March 19, 2026:
  - https://docs.comfy.org/registry/publishing
  - https://docs.comfy.org/registry/specifications
