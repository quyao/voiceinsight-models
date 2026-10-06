# VoiceInsight model artifacts

This public repository distributes versioned **VoiceInsight-maintained ONNX conversions of the author's DTLN-aec weights**. These are not author-published official ONNX files. No VoiceInsight application source, credentials, meeting recordings, transcripts or speaker embeddings are hosted here.

## Original model and license

- Author: Nils L. Westhausen, DTLN-aec.
- Upstream: https://github.com/breizhn/DTLN-aec
- Fixed upstream commit: `9d24e128b4f409db18227b8babb343016625921f`.
- Original repository license: MIT. Each release includes the original `LICENSE-DTLN-aec` and a `NOTICE` preserving attribution.
- 128, 256 and 512 refer to the LSTM size, not the audio sample rate or window size. Each model is a matched pair of neural stages; install both files from the same variant and release.

## Release layout

Tags: `aec-dtln-128-v1.0.0`, `aec-dtln-256-v1.0.0`, `aec-dtln-512-v1.0.0`.

Each pre-release contains:

- `model_<size>_1.onnx` and `model_<size>_2.onnx`
- `manifest.json`: source URLs/hashes, tool recipe, input/output contract and artifact hashes
- `LICENSE-DTLN-aec`, `NOTICE`
- `validation-summary.json`: mandatory ONNX gates and separately reported cross-runtime diagnostics

Assets are version-pinned. Do not mix sizes/stages or treat `latest` as a model selection policy. A missing or damaged selected model must be an error, not an implicit switch to another size.

## Conversion recipe

Maintainer build: macOS arm64, Python 3.11.15, TensorFlow 2.16.2, tf2onnx 1.16.1, ONNX 1.17.0, ONNX Runtime 1.20.1, NumPy 1.26.4. Complete resolved package versions and source SHA-256 are recorded with each release.

For each author's fixed TFLite stage, the conversion command is:

```text
python -m tf2onnx.convert --tflite dtln_aec_<size>_<stage>.tflite --opset 18 --output model_<size>_<stage>.onnx
```

All model weights/graph topology, tensor interfaces, finite non-silent inference and explicit recurrent-state reset were checked. Generated internal node names can affect serialized ONNX SHA across independent conversions: published assets have fixed hashes, while independent rebuild equivalence must also consider graph/weight identity. The pipeline does not train new weights or depend on anarlog artifacts.

## Maintainer tooling in this repository

This repository owns the whole TFLite-to-ONNX pipeline: conversion, artifact gates, independent re-verification and Draft-only publishing. The VoiceInsight application repository contains no conversion code and no Python environment; it only downloads these published artifacts through its native Rust downloader.

```text
tools/aec-model/   builder.py, release.mjs, spec.json, pyproject.toml, uv.lock, license, docs
scripts/aec-model.mjs   local CLI entry (build / verify / publish)
tests/aec-model.test.mjs   Node regression (npm test, node --test)
```

```bash
npm test                                                  # Node regression, no network, no model needed
npm run model:aec:build -- --variant 256 --version 1.0.0
npm run model:aec:verify -- --variant 256 --version 1.0.0 --directory /absolute/path/to/bundle
npm run model:aec:publish -- --variant 256 --version 1.0.0 --directory /absolute/path/to/bundle --repo quyao/voiceinsight-models --target <40-hex-commit> --dry-run
```

`build` requires macOS arm64 and uv 0.12.22 (it creates a locked private Python 3.11 environment under `tools/aec-model/.venv`); `verify` and `publish` reuse that environment and never download or re-convert models. Every run writes `test-results/aec-model-<time>-<uuid>/` and nothing is overwritten. Publishing creates a new **Draft** release only, requires the explicit `--confirm-draft --acknowledge-limitations` confirmations, verifies remote asset size/SHA-256, and never makes a release public, overwrites an existing tag/release or runs retries-to-pass. There are no GitHub Actions or remote builders. `spec.json`, `builder.py`, `pyproject.toml` and `uv.lock` are hashed into each published `manifest.json`, so these four files must stay byte-identical; `verify` rejects a bundle if they change. Full details and trust boundaries: [`tools/aec-model/README.md`](tools/aec-model/README.md), execution record: [`tools/aec-model/VALIDATION.md`](tools/aec-model/VALIDATION.md).

## Evaluation boundary and known failures

2026-10-05 local comparison on Apple M1 Max: 12 fixed, SHA-checked public-English diagnostic scenarios; 3 fresh-state repetitions per size, the same production neural alignment/reset/priming, Silero/SenseVoice path and unchanged quality thresholds. Includes near/quiet speech, double talk, synthetic delayed/nonlinear echoes and two measured Surrey room impulse responses. No live microphone capture. These are already-seen diagnostic clips, not unseen-speaker validation.

| Size | Cases passing quality | Median processing RTF | Median DSP-process peak RSS | ONNX pair size |
|---|---:|---:|---:|---:|
| 128 | 7/12 | 0.061 | 48 MiB | 6.95 MiB |
| 256 | 8/12 | 0.089 | 64 MiB | 14.84 MiB |
| 512 | 11/12 | 0.177 | 94 MiB | 39.61 MiB |

- **No variant passed every acoustic quality case.** All three failed the 200ms far-end-only echo scenario; 512 still has that limitation.
- 128 also failed far-only 0ms, nonlinear/noisy far-end, and both room far-only cases due to residual microphone final text (and, for some cases, insufficient ERLE).
- 256 also failed far-only 0ms, nonlinear/noisy far-end, and quiet double-talk WER degradation.
- 512 performed best in this small set, but is not guaranteed superior in every room, language or device.
- RTF excludes model loading, ASR, process startup and IO. These are optimized development-build measurements, not end-to-end/release latency claims. RSS includes runtime and PCM buffers, not weights alone.
- Strict TFLite-vs-ONNX recurrent-state diagnostics can report **FAIL** at the unchanged `1e-4` tolerance. These are retained in each validation report. ONNX packaging gates passing does not imply exact cross-runtime parity or approval to switch inference runtimes.

The models are published as **pre-releases with known limitations**, not a claim that all application, acoustic, hardware, security or deployment acceptance tests passed. See each manifest/validation report before use.
