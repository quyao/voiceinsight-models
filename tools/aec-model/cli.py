#!/usr/bin/env python3
"""Standalone maintainer entry: build, verify and Draft-publish DTLN-aec ONNX artifacts.

Run from this repository::

    python3 tools/aec-model/cli.py build   --variant 256 --version 1.0.0
    python3 tools/aec-model/cli.py verify  --variant 256 --version 1.0.0 --directory <bundle>
    python3 tools/aec-model/cli.py publish --variant 256 --version 1.0.0 --directory <bundle> \
        --repo owner/name --target <40-hex-commit> --dry-run

Only the standard library runs this file; the model work happens in the pinned
``tools/aec-model/.venv`` created by ``uv sync --locked``. Building requires
macOS arm64 and uv 0.12.22; verify and publish never download or re-convert
models. Dry-run performs zero GitHub calls. Publishing creates a NEW DRAFT ONLY,
verifies remote asset hashes, and never overwrites or makes a release public.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from release import (  # noqa: E402  (path is set above)
    ToolError, github_client, inspect_bundle, json_read, json_write, publish_draft,
    run_process, safe_env, select_spec, tool_hashes, validate_repo, validate_target,
    validate_variant, validate_version,
)

PROJECT = Path(__file__).resolve().parents[2]
TOOL = Path(__file__).resolve().parent
PYTHON = TOOL / ".venv" / "bin" / "python"

USAGE = """DTLN-aec ONNX artifact maintenance (this repository only)
  python3 tools/aec-model/cli.py build   --variant 128|256|512 --version 1.0.0
  python3 tools/aec-model/cli.py verify  --variant 128|256|512 --version 1.0.0 --directory <bundle>
  python3 tools/aec-model/cli.py publish --variant 128|256|512 --version 1.0.0 --directory <bundle> \\
      --repo owner/name --target <40-hex-commit> --dry-run
  python3 tools/aec-model/cli.py publish --variant 128|256|512 --version 1.0.0 --directory <bundle> \\
      --repo owner/name --target <40-hex-commit> --confirm-draft --acknowledge-limitations
Build requires uv 0.12.22 and macOS arm64; installs a locked private Python environment.
Verify/publish require that environment, never download/convert models. Dry-run has ZERO GitHub calls.
Publish creates a NEW DRAFT ONLY, checks remote asset hashes, never overwrites or makes public.
"""

VALUE_FLAGS = {"--variant": "variant", "--version": "version", "--directory": "directory",
               "--repo": "repo", "--target": "target"}
BOOL_FLAGS = {"--dry-run": "dry_run", "--confirm-draft": "confirm_draft",
              "--acknowledge-limitations": "acknowledge_limitations"}


class Blocked(ToolError):
    """Another run owns the workspace lock; do not remove or steal it."""


def parse(argv):
    if "--help" in argv or not argv:
        return {"help": True}
    action = argv[0]
    if action not in ("build", "verify", "publish"):
        raise ToolError("choose build, verify or publish")
    options = {"action": action}
    rest = argv[1:]
    index = 0
    while index < len(rest):
        token = rest[index]
        key = VALUE_FLAGS.get(token) or BOOL_FLAGS.get(token)
        if not key or key in options:
            raise ToolError("unknown or duplicate option: {0}".format(token))
        if token in VALUE_FLAGS:
            value = rest[index + 1] if index + 1 < len(rest) else None
            if value is None or value.startswith("--"):
                raise ToolError("option value missing for {0}".format(token))
            options[key] = value
            index += 1
        else:
            options[key] = True
        index += 1
    validate_version(options.get("version"))
    validate_variant(options.get("variant"))
    if action != "build":
        if not options.get("directory"):
            raise ToolError("--directory is required")
        options["directory"] = str(Path(options["directory"]).resolve())
    elif options.get("directory"):
        raise ToolError("build output is always a fresh test-results run in this repository; "
                        "no arbitrary overwrite")
    if action == "publish":
        validate_repo(options.get("repo"))
        validate_target(options.get("target"))
        if options.get("dry_run") and options.get("confirm_draft"):
            raise ToolError("dry-run and confirm-draft are mutually exclusive")
        if not options.get("dry_run") and not (options.get("confirm_draft")
                                               and options.get("acknowledge_limitations")):
            raise ToolError("choose --dry-run or both --confirm-draft --acknowledge-limitations")
    else:
        for key in ("repo", "target", "dry_run", "confirm_draft", "acknowledge_limitations"):
            if options.get(key):
                raise ToolError("publish flags only apply to publish")
    return options


def run_id(variant):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3] + "Z"
    return "aec-model-{0}-{1}-{2}".format(variant, stamp, uuid.uuid4())


def main(argv=None):
    options = parse(sys.argv[1:] if argv is None else list(argv))
    if options.get("help"):
        print(USAGE)
        return 0

    spec = select_spec(json_read(TOOL / "spec.json"), options["variant"])
    output_root = PROJECT / "test-results"
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / run_id(options["variant"])
    run.mkdir(mode=0o700)
    lock = output_root / ".model-tool-lock"
    abort = threading.Event()
    summary = {"action": options["action"], "variant": options["variant"], "version": options["version"],
               "status": "RUNNING", "fullAcceptancePassed": False, "applicationSourceTouched": False,
               "scope": "standalone model artifact repository only"}
    owns_lock = False
    write_summary = lambda: json_write(run / "summary.json", summary)

    try:
        try:
            os.mkdir(lock, 0o700)
        except FileExistsError:
            raise Blocked("Another run owns {0}. Do not run builds/publishes concurrently.".format(lock))
        owns_lock = True
        json_write(lock / "owner.json", {"pid": os.getpid(), "startedAt": datetime.now(timezone.utc).isoformat(),
                                         "runId": str(run), "tool": "aec-model"})
        for signame in ("SIGINT", "SIGTERM"):
            try:
                import signal as signal_module
                signal_module.signal(getattr(signal_module, signame), lambda *_: abort.set())
            except (ImportError, AttributeError, ValueError):
                pass

        before = tool_hashes(TOOL)

        def verification(directory, version):
            if not PYTHON.exists():
                raise ToolError("locked environment missing; run build first (uv sync --locked)")
            result = run_process(str(PYTHON), [str(TOOL / "builder.py"), "verify", "--variant", options["variant"],
                                               "--version", version, "--directory", str(directory)],
                                 cwd=str(PROJECT), env=safe_env(), abort=abort, timeout_ms=180_000)
            (run / "verification.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.code != 0:
                raise ToolError("ONNX verification failed; inspect local verification.log")

        if options["action"] == "build":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise ToolError("v1 builder requires macOS arm64")
            version = run_process("uv", ["--version"], env=safe_env(), abort=abort, timeout_ms=10_000)
            found = version.stdout.strip().split(" ")[1] if version.stdout.strip().startswith("uv ") else None
            if version.code != 0 or found != spec["recipe"]["uv"]:
                raise ToolError("requires uv {0}".format(spec["recipe"]["uv"]))
            sync = run_process("uv", ["sync", "--locked", "--no-dev", "--no-build", "--no-config",
                                      "--default-index", "https://pypi.org/simple",
                                      "--project", str(TOOL), "--python", spec["recipe"]["python"]],
                               cwd=str(PROJECT), env=safe_env(), abort=abort, timeout_ms=900_000)
            (run / "environment-setup.log").write_text(sync.stdout + sync.stderr, encoding="utf-8")
            if sync.code != 0:
                raise ToolError("locked environment setup failed; inspect local environment-setup.log")
            built = run_process(str(PYTHON), [str(TOOL / "builder.py"), "build", "--variant", options["variant"],
                                              "--version", options["version"], "--directory", str(run)],
                                cwd=str(PROJECT), env=safe_env(), abort=abort, timeout_ms=600_000)
            (run / "build.log").write_text(built.stdout + built.stderr, encoding="utf-8")
            if built.code != 0:
                raise ToolError("artifact build failed; inspect local build.log "
                                "(partial files are not a release)")
            directory = run / "bundle"
            bundle = inspect_bundle(directory, options["version"], spec, before)
            summary.update(status="PASS", directory=str(directory), mandatoryGatesPassed=True,
                           diagnostics=bundle["validation"]["crossRuntimeDiagnostics"], published=False,
                           next="Inspect diagnostics; use explicit publish --dry-run before any Draft upload.")
        elif options["action"] == "verify":
            bundle = inspect_bundle(options["directory"], options["version"], spec, before)
            verification(options["directory"], options["version"])
            after = inspect_bundle(options["directory"], options["version"], spec, before)
            if bundle["files"] != after["files"]:
                raise ToolError("bundle changed during verification")
            summary.update(status="PASS", mandatoryGatesPassed=True,
                           diagnostics=after["validation"]["crossRuntimeDiagnostics"], published=False)
        else:
            published = publish_draft(options, spec=spec, expected_tools=before, verify=verification,
                                      client=github_client(abort=abort), work=run,
                                      on_created=lambda info: (summary.update(createdDraft=info), write_summary()))
            summary.update(status="PASS", publication=published)

        if tool_hashes(TOOL) != before:
            raise ToolError("tool source changed during execution")
    except ToolError as error:
        blocked = isinstance(error, Blocked)
        summary.update(status="BLOCKED" if blocked else "FAIL", error=str(error))
        exit_code = 2 if blocked else 1
    else:
        exit_code = 0
    finally:
        if owns_lock:
            try:
                owner = json_read(lock / "owner.json")
                if owner.get("pid") != os.getpid():
                    raise ToolError("workspace lock ownership changed")
                shutil.rmtree(lock)
                summary["cleanup"] = "PASS"
            except (OSError, ValueError, ToolError):
                summary["cleanup"] = "FAIL"
                summary["status"] = "FAIL"
                exit_code = 1
        write_summary()
        print("AEC model {0}: {1}\nReport: {2}".format(options["action"], summary["status"], run / "summary.json"))
        if summary.get("directory"):
            print("Bundle: {0}".format(summary["directory"]))
        if summary.get("error"):
            print(summary["error"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ToolError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
