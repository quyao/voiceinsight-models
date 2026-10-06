"""Artifact inspection, publish planning and Draft upload for DTLN-aec ONNX releases.

Pure standard library port of the previous ``release.mjs``. Nothing here is allowed
to change ``spec.json``, ``builder.py``, ``pyproject.toml`` or ``uv.lock``: the
SHA-256 of those four files is recorded as ``toolHashes`` in every published
manifest and re-checked on verify, so this module and ``cli.py`` stay separate
files that use only the standard library and the already locked Python packages.

Trust boundaries kept from the Node implementation:

* a bundle is a fixed six-file allowlist of regular files (no symlinks, bounded
  size, content hashes re-derived locally);
* dry-run performs zero network calls; ``gh`` is never started before the
  snapshot has been re-verified;
* credentials are only ever handed to the ``gh`` child through an allowlisted
  environment, never through argv, files or logs;
* publishing creates a new Draft only, addressed by its release ID, never
  overwrites, never retries to pass and never makes a release public.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal as signal_module
import stat as stat_module
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import quote

TOOL_FILES = ("spec.json", "builder.py", "pyproject.toml", "uv.lock")
GATES = ("sourceHashes", "authorGraphAndWeights", "interfaceContract", "finiteNonSilentInference", "recurrentReset")
UPSTREAM_RAW = "https://raw.githubusercontent.com/breizhn/DTLN-aec/{commit}/{path}"
MAX_ASSET_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_OUTPUT = 8 * 1024 * 1024
READ_CHUNK = 65536


class ToolError(RuntimeError):
    """Operator-facing failure; messages never carry child stderr or secrets."""


def require(condition, message):
    if not condition:
        raise ToolError(message)


# --------------------------------------------------------------------------- spec


def validate_variant(value):
    require(value in ("128", "256", "512"), "--variant must explicitly be 128, 256 or 512")
    return value


def assets_for(variant):
    validate_variant(variant)
    return sorted(["LICENSE-DTLN-aec", "NOTICE", "manifest.json",
                   "model_{0}_1.onnx".format(variant), "model_{0}_2.onnx".format(variant),
                   "validation-summary.json"])


def select_spec(base, variant):
    validate_variant(variant)
    require(isinstance(base, dict), "spec must be an object")
    variants = base.get("variants")
    require(isinstance(variants, dict) and variants.get(variant), "variant recipe is missing")
    common = {key: value for key, value in base.items() if key not in ("variants", "licenseSource")}
    choice = variants[variant]
    return dict(common, id="dtln-aec-{0}".format(variant), variant=variant,
                contract=dict(common["contract"], stateShape=[1, 2, int(variant), 2]),
                sources=[*choice["sources"], base["licenseSource"]],
                expectedGraphs=choice["expectedGraphs"])


_VERSION = re.compile(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}(?:-[a-z0-9]+(?:[.-][a-z0-9]+)*)?")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_TARGET = re.compile(r"[a-f0-9]{40}\Z")


def validate_version(value):
    require(isinstance(value, str) and value == value.strip() and len(value) <= 64
            and _VERSION.fullmatch(value) is not None, "invalid version")
    return value


def validate_repo(value):
    require(isinstance(value, str) and value == value.strip() and len(value) <= 200
            and _REPO.match(value) is not None, "expected GitHub owner/repository")
    return value


def validate_target(value):
    require(isinstance(value, str) and len(value) == 40 and _TARGET.fullmatch(value) is not None,
            "--target must be an explicit full Git commit SHA")
    return value


# ------------------------------------------------------------------------ basics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tool_hashes(root):
    root = Path(root)
    return {name: sha((root / name).read_bytes()) for name in TOOL_FILES}


def safe_env(source=None, github=False):
    source = os.environ if source is None else source
    env = {key: source[key] for key in ("PATH", "HOME", "TMPDIR", "SystemRoot") if source.get(key)}
    env.update({"TF_CPP_MIN_LOG_LEVEL": "2", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1"})
    if github:
        for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"):
            if source.get(key):
                env[key] = source[key]
        env.update({"GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1", "GIT_TERMINAL_PROMPT": "0",
                    "GH_PAGER": "cat", "NO_COLOR": "1"})
    return env


class ProcResult(tuple):
    """(code, signal, stdout, stderr) — never carries child stderr in errors."""

    def __new__(cls, code, signal, stdout, stderr):
        return super().__new__(cls, (code, signal, stdout, stderr))

    code = property(lambda self: self[0])
    signal = property(lambda self: self[1])
    stdout = property(lambda self: self[2])
    stderr = property(lambda self: self[3])


def _kill_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def run_process(command, args, *, cwd=None, env=None, input=None, timeout_ms=120_000,
                abort=None, max_bytes=DEFAULT_MAX_OUTPUT):
    """Bounded process group runner: timeout, output cap, no leaked descendants.

    The child leads its own session, so interrupting or finishing an invocation
    can only ever reap this invocation's descendants. Child stderr is returned to
    the caller but never embedded in raised messages.
    """
    env = safe_env() if env is None else env
    try:
        proc = subprocess.Popen([command, *args], cwd=cwd, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError:
        raise ToolError("cannot start {0}".format(command))

    lock = threading.Lock()
    state = {"stopped": None}
    total = [0]
    chunks = {"out": [], "err": []}
    done = threading.Event()
    escalation = []

    def kill(sig):
        _kill_group(proc.pid, sig)

    def stop(reason):
        with lock:
            if state["stopped"]:
                return
            state["stopped"] = reason
        kill(signal_module.SIGTERM)
        timer = threading.Timer(1.5, lambda: kill(signal_module.SIGKILL))
        timer.daemon = True
        timer.start()
        escalation.append(timer)

    def collect(stream, target):
        def reader():
            while True:
                try:
                    chunk = os.read(stream.fileno(), READ_CHUNK)
                except (OSError, ValueError):
                    return
                if not chunk:
                    return
                with lock:
                    total[0] += len(chunk)
                    exceeded = total[0] > max_bytes
                    if not exceeded:
                        chunks[target].append(chunk)
                if exceeded:
                    stop("output limit exceeded")
                    return
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        return thread

    def feed():
        try:
            if input:
                data = input.encode("utf-8") if isinstance(input, str) else input
                proc.stdin.write(data)
            proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def supervise():
        deadline = time.monotonic() + timeout_ms / 1000
        while not done.wait(0.05):
            if abort is not None and abort.is_set():
                stop("interrupted")
                return
            if time.monotonic() >= deadline:
                stop("timed out")
                return

    readers = [collect(proc.stdout, "out"), collect(proc.stderr, "err")]
    threading.Thread(target=feed, daemon=True).start()
    supervisor = threading.Thread(target=supervise, daemon=True)
    supervisor.start()
    try:
        proc.wait()
    finally:
        done.set()
        # Descendants may outlive a successful parent with inherited stdio closed.
        kill(signal_module.SIGKILL)
        for thread in readers:
            thread.join(timeout=5)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        for timer in escalation:
            timer.cancel()

    if state["stopped"]:
        raise ToolError("{0}: {1}".format(command, state["stopped"]))
    code = proc.returncode
    return ProcResult(code,
                      -code if code is not None and code < 0 else None,
                      b"".join(chunks["out"]).decode("utf-8", "replace"),
                      b"".join(chunks["err"]).decode("utf-8", "replace"))


# ------------------------------------------------------------------ bundle checks


def _regular_file(path):
    info = os.lstat(path)
    return stat_module.S_ISREG(info.st_mode), info.st_size


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def inspect_bundle(directory, version, spec, expected_tools):
    validate_version(version)
    assets = assets_for(spec["variant"])
    directory = Path(directory)
    try:
        info = os.lstat(directory)
    except OSError:
        raise ToolError("bundle directory is missing or unreadable")
    require(stat_module.S_ISDIR(info.st_mode) and not stat_module.S_ISLNK(info.st_mode),
            "bundle must be a real directory")
    try:
        names = sorted(entry.name for entry in os.scandir(directory))
    except OSError:
        raise ToolError("bundle directory is missing or unreadable")
    require(names == assets, "bundle must contain exactly the six public release assets")

    files = {}
    for name in names:
        path = directory / name
        try:
            regular, size = _regular_file(path)
            require(regular and not os.path.islink(path) and 0 < size <= MAX_ASSET_BYTES,
                    "invalid asset: {0}".format(name))
            data = path.read_bytes()
        except OSError:
            raise ToolError("invalid asset: {0}".format(name))
        files[name] = {"size": len(data), "sha256": sha(data)}

    try:
        manifest = json_read(directory / "manifest.json")
        validation = json_read(directory / "validation-summary.json")
    except (OSError, ValueError):
        raise ToolError("bundle manifest/validation report is not readable JSON")
    require(manifest.get("schemaVersion") == 1 and manifest.get("model") == spec["id"]
            and manifest.get("variant") == spec["variant"] and manifest.get("version") == version,
            "manifest identity mismatch")
    for key in ("upstream", "recipe", "contract"):
        require(manifest.get(key) == spec[key], "manifest {0} mismatch".format(key))
    sources = [dict(entry, url=UPSTREAM_RAW.format(commit=spec["upstream"]["commit"], path=entry["path"]))
               for entry in spec["sources"]]
    require(manifest.get("sources") == sources, "unexpected upstream source")
    require(manifest.get("toolHashes") == expected_tools and validation.get("toolHashes") == expected_tools,
            "tool recipe changed; use the exact builder revision")
    require(sorted(manifest.get("files", {})) == [name for name in assets if name != "manifest.json"],
            "manifest asset allowlist mismatch")
    for name, expected in manifest["files"].items():
        require(expected == files[name], "checksum mismatch: {0}".format(name))
    require(files["LICENSE-DTLN-aec"] == {"size": spec["sources"][2]["size"], "sha256": spec["sources"][2]["sha256"]},
            "license mismatch")
    require(validation.get("schemaVersion") == 1 and validation.get("model") == spec["id"]
            and validation.get("variant") == spec["variant"] and validation.get("version") == version,
            "validation identity mismatch")
    gates = validation.get("gates") or {}
    require(validation.get("mandatoryGatesPassed") is True and sorted(gates) == sorted(GATES)
            and all(value is True for value in gates.values()), "mandatory validation did not pass")
    models = validation.get("models") or []
    diagnostics = validation.get("crossRuntimeDiagnostics") or []
    require(len(models) == 2 and len(diagnostics) == 2 and len(validation.get("limitations") or []) >= 3,
            "incomplete validation report")
    for index in range(2):
        model = models[index]
        name = "model_{0}_{1}.onnx".format(spec["variant"], index + 1)
        require(model.get("file") == name and model.get("sha256") == files[name]["sha256"]
                and model.get("size") == files[name]["size"]
                and model.get("graphFingerprint") == spec["expectedGraphs"][index]
                and _finite(model.get("resetMaxAbsoluteError")) and 0 <= model["resetMaxAbsoluteError"] <= 1e-6
                and model.get("framesPerPass") == 120 and model.get("passes") == 2,
                "model validation is not bound to these artifacts")
        diagnostic = diagnostics[index]
        require(diagnostic.get("tolerance") == spec["crossRuntimeTolerance"]
                and _finite(diagnostic.get("outputMaxError")) and diagnostic["outputMaxError"] >= 0
                and _finite(diagnostic.get("stateMaxError")) and diagnostic["stateMaxError"] >= 0,
                "invalid diagnostic metrics")
        worst = max(diagnostic["outputMaxError"], diagnostic["stateMaxError"])
        require(diagnostic.get("status") == ("PASS" if worst <= spec["crossRuntimeTolerance"] else "FAIL"),
                "false diagnostic verdict")
    return {"manifest": manifest, "validation": validation, "files": files}


def make_plan(options, bundle):
    validate_repo(options["repo"])
    validate_version(options["version"])
    validate_target(options["target"])
    variant = validate_variant(bundle["manifest"]["variant"])
    return {"repository": options["repo"], "variant": variant, "tag": "aec-dtln-{0}-v{1}".format(variant, options["version"]),
            "targetCommit": options["target"], "draft": True, "makeLatest": False,
            "assets": [dict(bundle["files"][name], name=name) for name in assets_for(variant)],
            "diagnostics": bundle["validation"]["crossRuntimeDiagnostics"],
            "limitations": bundle["validation"]["limitations"],
            "notice": "VoiceInsight-maintained ONNX conversion of author weights, NOT official author-published ONNX. "
                      "No full acoustic/runtime-switch acceptance."}


# ---------------------------------------------------------------------- GitHub


def parse_api_response(result):
    """``gh api --include`` prints HTTP headers even for non-2xx; fail closed."""
    normalized = result.stdout.replace("\r\n", "\n")
    boundary = normalized.find("\n\n")
    first = normalized.split("\n")[0]
    match = re.match(r"HTTP/\S+ ([0-9]{3})", first)
    require(match is not None and boundary >= 0,
            "GitHub response lacks an HTTP status; authentication/network failure")
    status = int(match.group(1))
    payload = normalized[boundary + 2:]
    try:
        body = json.loads(payload) if payload.strip() else None
    except ValueError:
        raise ToolError("invalid GitHub JSON response")
    require(result.code == 0 or status >= 400, "GitHub command failed")
    return {"status": status, "body": body}


def github_client(run=run_process, abort=None):
    def api(method, path, data=None):
        args = ["api", "--hostname", "github.com", "--include", "--method", method, path]
        if data is not None:
            args += ["--input", "-"]
        result = run("gh", args, env=safe_env(github=True),
                     input=None if data is None else json.dumps(data, separators=(",", ":")),
                     timeout_ms=60_000, abort=abort)
        return parse_api_response(result)

    def upload(repo, release_id, path):
        # Address the exact created ID, not a tag that another process could rebind.
        url = "https://uploads.github.com/repos/{0}/releases/{1}/assets?name={2}".format(
            repo, release_id, quote(Path(path).name, safe=""))
        result = run("gh", ["api", "--hostname", "github.com", "--method", "POST", url,
                            "--header", "Content-Type: application/octet-stream", "--input", path],
                     env=safe_env(github=True), timeout_ms=120_000, abort=abort)
        require(result.code == 0, "GitHub upload failed; partial Draft retained; no overwrite/retry")

    return {"api": api, "upload": upload}


# --------------------------------------------------------------------- publish


def publish_draft(options, *, spec, expected_tools, verify, client, work, on_created=None):
    """Snapshot, re-verify, then create a new Draft and upload the six assets.

    ``options`` keys: directory, version, repo, target, dry_run, confirm_draft,
    acknowledge_limitations. Dry-run returns before any credential or network use.
    """
    assets = assets_for(spec["variant"])
    initial = inspect_bundle(options["directory"], options["version"], spec, expected_tools)
    plan = make_plan(options, initial)
    if not options.get("dry_run"):
        require(options.get("confirm_draft"), "publishing requires --confirm-draft")
        require(options.get("acknowledge_limitations"),
                "publishing requires --acknowledge-limitations (quality/cross-runtime boundaries)")
    snapshot = Path(work) / "upload-snapshot"
    snapshot.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        for name in assets:
            shutil.copyfile(Path(options["directory"]) / name, snapshot / name)
        frozen = inspect_bundle(snapshot, options["version"], spec, expected_tools)
        require(initial["files"] == frozen["files"], "bundle changed while snapshotting")
        verify(snapshot, options["version"])  # real ONNX graph + inference, never a JSON flag
        require(inspect_bundle(snapshot, options["version"], spec, expected_tools)["files"] == frozen["files"],
                "snapshot changed during verification")
        for name in assets:
            os.chmod(snapshot / name, 0o400)
        if options.get("dry_run"):
            return {"status": "DRY_RUN", "networkCalls": 0, "plan": plan}
        # No gh call above this line: dry-run never touches credentials or network.
        repo = client["api"]("GET", "repos/{0}".format(options["repo"]))
        require(repo["status"] == 200 and (repo["body"] or {}).get("permissions", {}).get("push") is True,
                "GitHub repository unavailable or lacks push permission")
        commit = client["api"]("GET", "repos/{0}/commits/{1}".format(options["repo"], options["target"]))
        require(commit["status"] == 200 and (commit["body"] or {}).get("sha") == options["target"],
                "target commit is not available in this repository")
        release = client["api"]("GET", "repos/{0}/releases/tags/{1}".format(options["repo"], plan["tag"]))
        require(release["status"] == 404, "release already exists; refusing overwrite/resume"
                if release["status"] == 200 else "cannot verify release absence")
        tag = client["api"]("GET", "repos/{0}/git/ref/tags/{1}".format(options["repo"], plan["tag"]))
        require(tag["status"] == 404, "tag already exists; refusing reuse"
                if tag["status"] == 200 else "cannot verify tag absence")
        created = client["api"]("POST", "repos/{0}/releases".format(options["repo"]), {
            "tag_name": plan["tag"], "target_commitish": options["target"],
            "name": "DTLN-aec {0} ONNX {1}".format(spec["variant"], options["version"]),
            "body": "{0}\n\nMandatory ONNX packaging gates passed. Inspect validation-summary.json for "
                    "cross-runtime diagnostics (including FAIL) and limitations.\n\n{1}".format(
                        plan["notice"], "\n".join(plan["limitations"])),
            "draft": True, "prerelease": "-" in options["version"], "make_latest": "false"})
        body = created["body"] or {}
        require(created["status"] == 201 and isinstance(body.get("id"), int) and not isinstance(body.get("id"), bool)
                and body.get("draft") is True and body.get("tag_name") == plan["tag"],
                "unexpected Draft creation response; inspect GitHub before retrying")
        release_id = body["id"]
        try:
            if on_created:
                on_created({"id": release_id, "repository": options["repo"], "tag": plan["tag"], "draft": True})
            for name in assets:
                require(inspect_bundle(snapshot, options["version"], spec, expected_tools)["files"] == frozen["files"],
                        "snapshot changed before upload")
                # Use the release ID for state checks; never upload to a published release.
                state = client["api"]("GET", "repos/{0}/releases/{1}".format(options["repo"], release_id))
                state_body = state["body"] or {}
                require(state["status"] == 200 and state_body.get("id") == release_id
                        and state_body.get("draft") is True and state_body.get("tag_name") == plan["tag"],
                        "Draft changed externally; stop uploading")
                client["upload"](options["repo"], release_id, str(snapshot / name))
            state = client["api"]("GET", "repos/{0}/releases/{1}".format(options["repo"], release_id))
            state_body = state["body"] or {}
            require(state["status"] == 200 and state_body.get("draft") is True
                    and state_body.get("id") == release_id and state_body.get("tag_name") == plan["tag"],
                    "Draft changed externally after upload")
            remote = client["api"]("GET", "repos/{0}/releases/{1}/assets?per_page=100".format(options["repo"], release_id))
            remote_body = remote["body"]
            require(remote["status"] == 200 and isinstance(remote_body, list) and len(remote_body) == len(assets),
                    "remote asset list mismatch")
            require(sorted(asset.get("name") for asset in remote_body) == assets, "remote asset names mismatch")
            for asset in remote_body:
                expected = frozen["files"][asset.get("name")]
                require(asset.get("state") == "uploaded" and asset.get("size") == expected["size"]
                        and asset.get("digest") == "sha256:{0}".format(expected["sha256"]),
                        "remote checksum missing/mismatch: {0}".format(asset.get("name")))
            return {"status": "DRAFT_UPLOADED", "releaseId": release_id, "plan": plan,
                    "remoteHashesVerified": True, "publiclyPublished": False}
        except ToolError:
            # Never delete somebody else's release/tag or overwrite assets on retries.
            raise ToolError("Draft {0} release ID {1} retained after failure; inspect manually. "
                            "No automatic delete, retry or public publish.".format(options["repo"], release_id))
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)
