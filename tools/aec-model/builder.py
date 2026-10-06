"""Local maintainer tool. No application settings, installed models or anarlog inputs.
Build imports TensorFlow only for conversion/diagnostics. Verify imports ONNX/ORT,
never TensorFlow, never downloads, and never regenerates artifacts.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE_SPEC = json.loads((HERE / "spec.json").read_text())
SPEC = None
TOOL_FILES = ["spec.json", "builder.py", "pyproject.toml", "uv.lock"]
MODEL_NAMES = []
BUNDLE_FILES = []
VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_variant(variant):
    global SPEC, MODEL_NAMES, BUNDLE_FILES
    require(variant in ("128", "256", "512"), "unsupported AEC variant")
    choice = BASE_SPEC["variants"][variant]
    SPEC = {k: v for k, v in BASE_SPEC.items() if k not in ("variants", "licenseSource")}
    SPEC.update(id=f"dtln-aec-{variant}", variant=variant,
                contract={**BASE_SPEC["contract"], "stateShape": [1, 2, int(variant), 2]},
                sources=choice["sources"] + [BASE_SPEC["licenseSource"]], expectedGraphs=choice["expectedGraphs"])
    MODEL_NAMES = [f"model_{variant}_{part}.onnx" for part in (1, 2)]
    BUNDLE_FILES = sorted(MODEL_NAMES + ["LICENSE-DTLN-aec", "NOTICE", "validation-summary.json", "manifest.json"])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    require(path.is_file() and not path.is_symlink(), f"not a regular artifact: {path.name}")
    return {"size": path.stat().st_size, "sha256": digest(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tool_hashes():
    return {name: digest(HERE / name) for name in TOOL_FILES}


def check_environment():
    require(platform.python_version() == SPEC["recipe"]["python"], "unexpected Python version")
    require(platform.system() == SPEC["recipe"]["system"] and platform.machine() == SPEC["recipe"]["machine"],
            "v1 builder is validated only on macOS arm64; ONNX assets are platform independent")
    actual = {name: importlib.metadata.version(name) for name in SPEC["recipe"]["packages"]}
    require(actual == SPEC["recipe"]["packages"], "conversion/verification package versions do not match recipe")
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": dict(sorted((d.metadata["Name"].lower(), d.version) for d in importlib.metadata.distributions()))}


class PinnedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlparse(newurl)
        require(url.scheme == "https" and url.hostname == "raw.githubusercontent.com", "unexpected upstream redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_source(entry, directory):
    # One bounded request/file. No arbitrary URLs, credentials, retries or caches.
    url = f"https://raw.githubusercontent.com/breizhn/DTLN-aec/{SPEC['upstream']['commit']}/{entry['path']}"
    target = directory / entry["name"]
    with urllib.request.build_opener(PinnedRedirects()).open(url, timeout=60) as response:
        data = response.read(entry["size"] + 1)
    require(len(data) == entry["size"], f"source size mismatch: {entry['name']}")
    require(hashlib.sha256(data).hexdigest() == entry["sha256"], f"source checksum mismatch: {entry['name']}")
    target.write_bytes(data)
    return {**entry, "url": url}


def graph_fingerprint(model):
    """V1 fingerprint from the evaluated recipe: all weights/ops/attributes/topology.
    Ignore generated internal names and ordering of commutative operands. The
    public input/output names, shapes, dtypes and opsets are checked separately.
    """
    import onnx
    values = {"": "missing-optional-input"}
    for i, item in enumerate(model.graph.input):
        values[item.name] = f"input-{i}-" + str(item.type)
    for item in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(item)
        values[item.name] = hashlib.sha256(str((arr.shape, str(arr.dtype))).encode() + arr.tobytes()).hexdigest()
    for node in model.graph.node:
        attrs = [a.SerializeToString().hex() for a in sorted(node.attribute, key=lambda a: a.name)]
        inputs = [values[x] for x in node.input]
        if node.op_type in ("Add", "Mul", "Max", "Min"):
            inputs.sort()
        h = hashlib.sha256(json.dumps([node.domain, node.op_type, inputs, attrs]).encode()).hexdigest()
        for i, name in enumerate(node.output):
            values[name] = f"{h}:{i}"
    return [values[x.name] for x in model.graph.output]


def sequence(session, seed=20261005, count=120):
    import numpy as np
    rng = np.random.default_rng(seed)
    state = np.zeros(SPEC["contract"]["stateShape"], np.float32)
    result, inputs = [], []
    n = session.get_inputs()[0].shape[-1]
    for _ in range(count):
        x, r = [rng.normal(0, .2, (1, 1, n)).astype(np.float32) for _ in range(2)]
        if n == 257:
            x, r = abs(x), abs(r)
        values = session.run(None, dict(zip(SPEC["contract"]["inputNames"][0 if n == 257 else 1], (x, state, r))))
        require(all(np.isfinite(v).all() for v in values), "non-finite recurrent inference")
        inputs.append((x, r))
        result.append(values)
        state = values[1]
    require(max(float(np.max(abs(v[0]))) for v in result) > .005, "silent recurrent test cannot prove inference")
    return inputs, result


def validate_model(path, stage):
    import numpy as np
    import onnx
    import onnxruntime as ort
    model = onnx.load(str(path), load_external_data=False)
    # Self-contained assets only; reject external tensor files/path traversal.
    require(not any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in model.graph.initializer),
            "external ONNX tensor data is not allowed")
    onnx.checker.check_model(model)
    require({o.domain: o.version for o in model.opset_import} == {"": 18, "ai.onnx.ml": 2}, "unexpected opsets")
    fingerprint = graph_fingerprint(model)
    require(fingerprint == SPEC["expectedGraphs"][stage], "graph/weight fingerprint differs from evaluated author-derived model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    n = SPEC["contract"]["signalSizes"][stage]
    ins, outs = session.get_inputs(), session.get_outputs()
    require([x.name for x in ins] == SPEC["contract"]["inputNames"][stage], "input name mismatch")
    require([x.shape for x in ins] == [[1, 1, n], SPEC["contract"]["stateShape"], [1, 1, n]], "input shape mismatch")
    require([x.name for x in outs] == SPEC["contract"]["outputs"], "output name mismatch")
    require([x.shape for x in outs] == [[1, 1, n], SPEC["contract"]["stateShape"]], "output shape mismatch")
    require(all(x.type == "tensor(float)" for x in ins + outs), "tensor dtype mismatch")
    inputs, first = sequence(session)
    _, second = sequence(session)
    reset_error = max(float(np.max(abs(a-b))) for x,y in zip(first,second) for a,b in zip(x,y))
    require(reset_error <= 1e-6, "reset/recurrent output mismatch")
    return {"file": path.name, **record(path), "graphFingerprint": fingerprint, "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer), "framesPerPass": 120, "passes": 2,
            "resetMaxAbsoluteError": reset_error}, inputs, first


def cross_runtime_diagnostic(source, inputs, expected):
    # Deliberately NOT an ONNX release gate: this is a known failed strict
    # cross-runtime diagnostic, not proof of universal numerical/audio parity.
    import numpy as np
    import tensorflow as tf
    lite = tf.lite.Interpreter(model_path=str(source), num_threads=1)
    lite.allocate_tensors()
    ins, outs = lite.get_input_details(), lite.get_output_details()
    state = np.zeros(SPEC["contract"]["stateShape"], np.float32)
    errors = [0.0, 0.0]
    for (x,r), reference in zip(inputs, expected):
        for entry,value in zip(ins,(x,state,r)):
            lite.set_tensor(entry["index"], value)
        lite.invoke()
        values = [lite.get_tensor(entry["index"]) for entry in outs]
        for i,(a,b) in enumerate(zip(values,reference)):
            require(np.isfinite(a).all(), "non-finite TFLite diagnostic")
            errors[i] = max(errors[i], float(np.max(abs(a-b))))
        state = values[1]
    return {"status": "PASS" if max(errors) <= SPEC["crossRuntimeTolerance"] else "FAIL",
            "tolerance": SPEC["crossRuntimeTolerance"], "outputMaxError": errors[0], "stateMaxError": errors[1],
            "scope": "120 recurrent frames; independent ONNX and TFLite state; not a runtime switch approval"}


def build(version, root):
    require(re.fullmatch(VERSION, version) and len(version) <= 64, "invalid version")
    environment = check_environment()
    before = tool_hashes()
    require(not (root / "bundle").exists() and not (root / "staging").exists(), "refusing to overwrite build artifacts")
    source_dir, staging = root / "sources", root / "staging"
    source_dir.mkdir()
    staging.mkdir()
    sources = [download_source(entry, source_dir) for entry in SPEC["sources"]]
    rows, diagnostics = [], []
    for stage,name in enumerate(MODEL_NAMES):
        source = source_dir / sources[stage]["name"]
        with (root / f"conversion-{stage+1}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--tflite", str(source), "--opset", "18",
                            "--output", str(staging / name)], check=True, timeout=120, stdout=log, stderr=subprocess.STDOUT)
        row, inputs, expected = validate_model(staging / name, stage)
        rows.append(row)
        diagnostics.append(cross_runtime_diagnostic(source, inputs, expected))
    require(before == tool_hashes(), "tool source changed during build")
    validation = {"schemaVersion": 1, "model": SPEC["id"], "variant": SPEC["variant"], "version": version, "mandatoryGatesPassed": True,
                  "gates": {"sourceHashes": True, "authorGraphAndWeights": True, "interfaceContract": True,
                            "finiteNonSilentInference": True, "recurrentReset": True},
                  "models": rows, "crossRuntimeDiagnostics": diagnostics, "toolHashes": before, "environment": environment,
                  "scope": "Author-derived ONNX packaging gates only; not full application, acoustic/device or TFLite runtime acceptance",
                  "limitations": ["Existing acoustic failures are not repaired by repackaging.",
                                  "Cross-runtime diagnostics may FAIL; these failures are preserved, not relaxed.",
                                  "No new 12-case ASR/device/full-acceptance run is implied by this build."]}
    write_json(staging / "validation-summary.json", validation)
    shutil.copyfile(source_dir / "LICENSE-DTLN-aec", staging / "LICENSE-DTLN-aec")
    (staging / "NOTICE").write_text(
        "DTLN-aec by Nils L. Westhausen. Original repository licensed MIT; see LICENSE-DTLN-aec.\n"
        f"Upstream: {SPEC['upstream']['repository']} at {SPEC['upstream']['commit']}\n"
        "These ONNX files are VoiceInsight-maintained conversions, NOT official author-published ONNX.\n"
        "No new training, no claimed cure for acoustic quality failures. See validation-summary.json.\n")
    manifest = {"schemaVersion": 1, "model": SPEC["id"], "variant": SPEC["variant"], "version": version,
                "upstream": SPEC["upstream"], "sources": sources, "recipe": SPEC["recipe"], "contract": SPEC["contract"],
                "toolHashes": before, "files": {p.name: record(p) for p in sorted(staging.iterdir())}}
    write_json(staging / "manifest.json", manifest)
    verify(staging, version)
    require(not (root / "bundle").exists(), "output appeared during build")
    staging.rename(root / "bundle")
    print(f"BUILD PASS: {root / 'bundle'}; mandatory ONNX gates only; diagnostics remain explicit", flush=True)


def verify(directory, version):
    check_environment()  # Metadata lookup only; no TensorFlow import/download/conversion.
    require(re.fullmatch(VERSION, version) and len(version) <= 64, "invalid version")
    require(directory.is_dir() and not directory.is_symlink(), "bundle must be a real directory")
    require(sorted(p.name for p in directory.iterdir()) == BUNDLE_FILES, "unexpected or missing bundle files")
    for p in directory.iterdir():
        require(p.is_file() and not p.is_symlink() and 0 < p.stat().st_size <= 32*1024*1024, "invalid bundle entry")
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "validation-summary.json").read_text())
    require(manifest["schemaVersion"] == 1 and manifest["model"] == SPEC["id"] and manifest["variant"] == SPEC["variant"] and manifest["version"] == version, "manifest identity mismatch")
    for key in ("upstream", "recipe", "contract"):
        require(manifest[key] == SPEC[key], f"manifest {key} mismatch")
    expected_sources = [{**entry, "url": f"https://raw.githubusercontent.com/breizhn/DTLN-aec/{SPEC['upstream']['commit']}/{entry['path']}"} for entry in SPEC["sources"]]
    require(manifest["sources"] == expected_sources, "manifest source mismatch")
    require(manifest["toolHashes"] == report["toolHashes"] == tool_hashes(), "use the exact tool revision that built this bundle")
    require(set(manifest["files"]) == set(BUNDLE_FILES) - {"manifest.json"}, "manifest file allowlist mismatch")
    for name, expected in manifest["files"].items():
        require(record(directory / name) == expected, f"artifact checksum mismatch: {name}")
    require(record(directory / "LICENSE-DTLN-aec") == {"size": SPEC['sources'][2]['size'], "sha256": SPEC['sources'][2]['sha256']}, "license mismatch")
    require(report["schemaVersion"] == 1 and report["model"] == SPEC["id"] and report["variant"] == SPEC["variant"] and report["version"] == version, "validation identity mismatch")
    expected_gates = {"sourceHashes", "authorGraphAndWeights", "interfaceContract", "finiteNonSilentInference", "recurrentReset"}
    require(report["mandatoryGatesPassed"] is True and set(report["gates"]) == expected_gates and
            all(v is True for v in report["gates"].values()), "mandatory validation gates did not pass")
    require(len(report["models"]) == 2 and len(report["crossRuntimeDiagnostics"]) == 2, "incomplete validation")
    require(len(report["limitations"]) >= 3, "missing validation limitations")
    for stage, name in enumerate(MODEL_NAMES):
        row, _, _ = validate_model(directory / name, stage)
        require(row == report["models"][stage], "recorded model validation differs from actual artifact")
    for diagnostic in report["crossRuntimeDiagnostics"]:
        require(diagnostic["tolerance"] == SPEC["crossRuntimeTolerance"], "diagnostic tolerance changed")
        errors = [diagnostic["outputMaxError"], diagnostic["stateMaxError"]]
        require(all(isinstance(x, (int,float)) and math.isfinite(x) and x >= 0 for x in errors), "invalid diagnostic metrics")
        require(diagnostic["status"] == ("PASS" if max(errors) <= SPEC["crossRuntimeTolerance"] else "FAIL"), "false diagnostic verdict")
    print("VERIFY PASS: hashes, evaluated author graph/weights, interfaces and recurrent reset; no conversion", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--variant", required=True, choices=["128", "256", "512"])
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    select_variant(args.variant)
    if args.action == "build":
        build(args.version, args.directory)
    else:
        verify(args.directory, args.version)


if __name__ == "__main__":
    main()
