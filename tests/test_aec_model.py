"""Regression for the DTLN-aec artifact tooling (standard-library unittest).

Run from the repository root::

    python3 -m unittest discover -s tests -v

These tests never touch the network, GitHub, models or the private environment:
transport tests use fake clients and fake bundles. The real ONNX verifier is
exercised by ``cli.py verify`` against an actual published bundle.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "aec-model"
sys.path.insert(0, str(TOOL))

from release import (  # noqa: E402  (path is set above)
    GATES, ProcResult, ToolError, assets_for, github_client, inspect_bundle, parse_api_response,
    publish_draft, run_process, safe_env, select_spec, sha, tool_hashes, validate_repo,
    validate_target, validate_variant, validate_version,
)

BASE_SPEC = json.loads((TOOL / "spec.json").read_text(encoding="utf-8"))
SPEC = select_spec(BASE_SPEC, "256")
ASSETS = assets_for("256")
TOOLS = tool_hashes(TOOL)
VERSION = "1.0.0-test.1"
TARGET = "a" * 40


def put(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fixture(case, variant="256"):
    spec = select_spec(BASE_SPEC, variant)
    assets = assets_for(variant)
    root = Path(tempfile.mkdtemp(prefix="vi-aec-tool-"))
    case.addCleanup(shutil.rmtree, root, True)
    directory = root / "bundle"
    directory.mkdir()
    validation = {"schemaVersion": 1, "model": spec["id"], "variant": variant, "version": VERSION,
                  "mandatoryGatesPassed": True, "gates": {name: True for name in GATES}, "toolHashes": TOOLS,
                  "models": [], "crossRuntimeDiagnostics": [
                      {"status": "FAIL", "tolerance": spec["crossRuntimeTolerance"],
                       "outputMaxError": 0.0002, "stateMaxError": 0},
                      {"status": "PASS", "tolerance": spec["crossRuntimeTolerance"],
                       "outputMaxError": 0, "stateMaxError": 0}],
                  "limitations": ["not official ONNX", "not acoustic acceptance", "cross-runtime failure retained"]}
    for index in range(2):
        name = "model_{0}_{1}.onnx".format(variant, index + 1)
        # Fake bytes ONLY for transport/manifest unit tests; the real CLI always
        # calls the Python ONNX verifier before publishing.
        data = "not-a-real-model-{0}".format(index).encode()
        (directory / name).write_bytes(data)
        validation["models"].append({"file": name, "size": len(data), "sha256": sha(data),
                                     "graphFingerprint": spec["expectedGraphs"][index],
                                     "resetMaxAbsoluteError": 0, "framesPerPass": 120, "passes": 2})
    (directory / "LICENSE-DTLN-aec").write_bytes((TOOL / "DTLN-LICENSE").read_bytes())
    (directory / "NOTICE").write_text("VoiceInsight-maintained conversion, not official ONNX\n", encoding="utf-8")
    put(directory / "validation-summary.json", validation)
    manifest = {"schemaVersion": 1, "model": spec["id"], "variant": variant, "version": VERSION,
                "upstream": spec["upstream"], "recipe": spec["recipe"], "contract": spec["contract"],
                "toolHashes": TOOLS, "files": {},
                "sources": [dict(entry, url="https://raw.githubusercontent.com/breizhn/DTLN-aec/{0}/{1}".format(
                    spec["upstream"]["commit"], entry["path"])) for entry in spec["sources"]]}

    def refresh():
        for name in assets:
            if name == "manifest.json":
                continue
            data = (directory / name).read_bytes()
            manifest["files"][name] = {"size": len(data), "sha256": sha(data)}
        put(directory / "manifest.json", manifest)

    refresh()
    options = {"repo": "example/models", "variant": variant, "version": VERSION, "target": TARGET,
               "directory": str(directory), "confirm_draft": True, "acknowledge_limitations": True}
    calls, uploads = [], []
    client = {"api": None, "upload": None}

    def api(method, path, data=None):
        calls.append({"method": method, "path": path, "data": data})
        if path == "repos/example/models":
            return {"status": 200, "body": {"permissions": {"push": True}}}
        if "/commits/" in path:
            return {"status": 200, "body": {"sha": TARGET}}
        if "/releases/tags/" in path or "/git/ref/tags/" in path:
            return {"status": 404, "body": {}}
        if method == "POST":
            return {"status": 201, "body": {"id": 123, "draft": True, "tag_name": "aec-dtln-{0}-v{1}".format(variant, VERSION)}}
        if path.endswith("/assets?per_page=100"):
            bundle = inspect_bundle(directory, VERSION, spec, TOOLS)
            return {"status": 200, "body": [{"name": name, "size": bundle["files"][name]["size"], "state": "uploaded",
                                             "digest": "sha256:{0}".format(bundle["files"][name]["sha256"])}
                                            for name in assets]}
        return {"status": 200, "body": {"id": 123, "draft": True,
                                        "tag_name": "aec-dtln-{0}-v{1}".format(variant, VERSION)}}

    def upload(repo, release_id, path):
        uploads.append({"repo": repo, "id": release_id, "name": os.path.basename(path)})

    client["api"], client["upload"] = api, upload
    verifications = []

    def verify(path, version):
        case.assertNotEqual(Path(path), directory)
        case.assertEqual(sorted(os.listdir(path)), assets)
        verifications.append(path)

    deps = {"spec": spec, "expected_tools": TOOLS, "work": root, "client": client, "verify": verify}
    return {"root": root, "directory": directory, "spec": spec, "manifest": manifest, "validation": validation,
            "refresh": refresh, "options": options, "calls": calls, "uploads": uploads, "client": client,
            "deps": deps, "verified": lambda: verifications}


class ToolTest(unittest.TestCase):
    def test_rejects_traversal_implicit_targets_malformed_versions_and_repositories(self):
        for value in ["../1.0.0", "latest", "01.0.0", "1.0.0;id", "-x", "1.0.0\n"]:
            with self.assertRaises(ToolError):
                validate_version(value)
        for value in ["https://github.com/o/r", "../x", "o/r/extra", "--repo", "o/r?token=x"]:
            with self.assertRaises(ToolError):
                validate_repo(value)
        with self.assertRaises(ToolError):
            validate_target("main")
        self.assertEqual(validate_version(VERSION), VERSION)
        self.assertEqual(validate_target(TARGET), TARGET)

    def test_all_three_variants_have_disjoint_assets_tags_state_shape_and_reject_cross_variant_bundles(self):
        tags = set()
        for variant in ["128", "256", "512"]:
            state = fixture(self, variant)
            self.assertEqual(state["spec"]["contract"]["stateShape"], [1, 2, int(variant), 2])
            result = publish_draft(dict(state["options"], dry_run=True), **state["deps"])
            self.assertEqual(result["plan"]["variant"], variant)
            self.assertIn("model_{0}_1.onnx".format(variant), [asset["name"] for asset in result["plan"]["assets"]])
            self.assertEqual(result["plan"]["tag"], "aec-dtln-{0}-v{1}".format(variant, VERSION))
            tags.add(result["plan"]["tag"])
            wrong = select_spec(BASE_SPEC, "512" if variant == "128" else "128")
            with self.assertRaises(ToolError):
                inspect_bundle(state["directory"], VERSION, wrong, TOOLS)
        self.assertEqual(len(tags), 3)
        for invalid in [None, "64", "1024", 256, "../128", "128\n"]:
            with self.assertRaises(ToolError):
                validate_variant(invalid)

    def test_unreadable_bundle_is_a_clean_operator_error(self):
        with self.assertRaisesRegex(ToolError, "missing or unreadable"):
            inspect_bundle(Path(tempfile.gettempdir()) / "vi-aec-definitely-missing", VERSION, SPEC, TOOLS)

    def test_bundle_whitelist_and_sha_checks_reject_extra_files_and_tampering(self):
        state = fixture(self)
        inspect_bundle(state["directory"], VERSION, state["spec"], TOOLS)
        (state["directory"] / "token.txt").write_text("not-a-real-secret", encoding="utf-8")
        with self.assertRaisesRegex(ToolError, "six public"):
            inspect_bundle(state["directory"], VERSION, state["spec"], TOOLS)
        os.remove(state["directory"] / "token.txt")
        (state["directory"] / "model_256_1.onnx").write_text("corrupt", encoding="utf-8")
        with self.assertRaisesRegex(ToolError, "checksum"):
            inspect_bundle(state["directory"], VERSION, state["spec"], TOOLS)

    def test_rejects_symlink_assets_and_bundle_directory(self):
        state = fixture(self)
        alias = state["root"] / "alias"
        os.symlink(state["directory"], alias)
        with self.assertRaisesRegex(ToolError, "real directory"):
            inspect_bundle(alias, VERSION, state["spec"], TOOLS)
        path = state["directory"] / "NOTICE"
        os.remove(path)
        os.symlink(TOOL / "DTLN-LICENSE", path)
        with self.assertRaisesRegex(ToolError, "invalid asset"):
            inspect_bundle(state["directory"], VERSION, state["spec"], TOOLS)

    def test_rejects_false_validation_wrong_recipe_stale_tool_source_and_model_report_mismatch(self):
        def downgrade(state):
            state["validation"]["mandatoryGatesPassed"] = False

        def fake_diagnostic(state):
            state["validation"]["crossRuntimeDiagnostics"][0]["status"] = "PASS"

        def loose_tolerance(state):
            state["validation"]["crossRuntimeDiagnostics"][0]["tolerance"] = 1

        def fake_model_hash(state):
            state["validation"]["models"][0]["sha256"] = "f" * 64

        def moving_upstream(state):
            state["manifest"]["upstream"] = dict(state["spec"]["upstream"], commit="main")

        def stale_tools(state):
            state["manifest"]["toolHashes"] = dict(TOOLS, **{"builder.py": "f" * 64})

        for mutation in [downgrade, fake_diagnostic, loose_tolerance, fake_model_hash, moving_upstream, stale_tools]:
            state = fixture(self)
            mutation(state)
            put(state["directory"] / "validation-summary.json", state["validation"])
            state["refresh"]()
            with self.assertRaises(ToolError):
                inspect_bundle(state["directory"], VERSION, state["spec"], TOOLS)

    def test_dry_run_verifies_frozen_assets_preserves_fail_diagnostics_never_calls_github(self):
        state = fixture(self)
        result = publish_draft(dict(state["options"], dry_run=True, confirm_draft=False,
                                    acknowledge_limitations=False), **state["deps"])
        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(result["networkCalls"], 0)
        self.assertEqual(result["plan"]["diagnostics"][0]["status"], "FAIL")
        self.assertEqual(len(state["verified"]()), 1)
        self.assertEqual(state["calls"], [])
        self.assertEqual(state["uploads"], [])
        self.assertFalse((state["root"] / "upload-snapshot").exists())

    def test_live_publish_requires_explicit_draft_and_limitations_confirmation(self):
        for missing in ["confirm_draft", "acknowledge_limitations"]:
            state = fixture(self)
            with self.assertRaisesRegex(ToolError, "requires"):
                publish_draft(dict(state["options"], **{missing: False}), **state["deps"])
            self.assertEqual(state["calls"], [])

    def test_real_onnx_verifier_is_mandatory_even_when_json_gates_say_pass(self):
        state = fixture(self)

        def broken(path, version):
            raise ToolError("ONNX graph invalid")

        state["deps"]["verify"] = broken
        with self.assertRaisesRegex(ToolError, "ONNX graph invalid"):
            publish_draft(state["options"], **state["deps"])
        self.assertEqual(state["calls"], [])
        self.assertEqual(state["uploads"], [])

    def test_snapshot_modification_during_verification_aborts_before_github_even_in_dry_run(self):
        state = fixture(self)

        def mutate(path, version):
            (Path(path) / "NOTICE").write_text("changed", encoding="utf-8")

        state["deps"]["verify"] = mutate
        with self.assertRaisesRegex(ToolError, "checksum|snapshot changed"):
            publish_draft(dict(state["options"], dry_run=True), **state["deps"])
        self.assertEqual(state["calls"], [])

    def test_existing_release_tag_unknown_http_failures_and_permission_denial_never_mutate_github(self):
        for scenario in ["release", "tag", "auth", "permission", "target"]:
            state = fixture(self)
            original = state["client"]["api"]

            def guarded(method, path, data=None, _original=original, _scenario=scenario):
                result = _original(method, path, data)
                if _scenario == "permission" and path == "repos/example/models":
                    return {"status": 200, "body": {"permissions": {"push": False}}}
                if _scenario == "target" and "/commits/" in path:
                    return {"status": 404, "body": {}}
                if "/releases/tags/" in path and _scenario in ("release", "auth"):
                    return {"status": 200 if _scenario == "release" else 401, "body": {}}
                if _scenario == "tag" and "/git/ref/tags/" in path:
                    return {"status": 200, "body": {}}
                return result

            state["client"]["api"] = guarded
            with self.assertRaises(ToolError):
                publish_draft(state["options"], **state["deps"])
            self.assertFalse(any(call["method"] != "GET" for call in state["calls"]))
            self.assertEqual(state["uploads"], [])

    def test_successful_transport_uploads_only_six_verified_assets_to_exact_draft_id_never_publishes(self):
        state = fixture(self)
        created = []
        state["deps"]["on_created"] = created.append
        result = publish_draft(state["options"], **state["deps"])
        self.assertEqual(result["status"], "DRAFT_UPLOADED")
        self.assertFalse(result["publiclyPublished"])
        self.assertTrue(result["remoteHashesVerified"])
        self.assertEqual(created[0]["id"], 123)
        mutations = [call for call in state["calls"] if call["method"] != "GET"]
        self.assertEqual(len(mutations), 1)
        self.assertEqual(mutations[0]["method"], "POST")
        self.assertIs(mutations[0]["data"]["draft"], True)
        self.assertEqual(mutations[0]["data"]["make_latest"], "false")
        self.assertEqual(mutations[0]["data"]["target_commitish"], TARGET)
        self.assertEqual([entry["name"] for entry in state["uploads"]], ASSETS)
        self.assertTrue(all(entry["id"] == 123 for entry in state["uploads"]))

    def test_upload_failures_retain_draft_do_not_delete_retry_and_remove_local_snapshot(self):
        state = fixture(self)

        def failing(repo, release_id, path):
            raise ToolError("secret-provider-diagnostic")

        state["client"]["upload"] = failing
        with self.assertRaises(ToolError) as caught:
            publish_draft(state["options"], **state["deps"])
        self.assertIn("ID 123 retained", str(caught.exception))
        self.assertNotIn("secret-provider", str(caught.exception))
        self.assertFalse(any(call["method"] in ("DELETE", "PATCH") for call in state["calls"]))
        self.assertFalse((state["root"] / "upload-snapshot").exists())

    def test_remote_digest_failure_or_external_draft_state_change_cannot_report_success(self):
        for scenario in ["digest", "published"]:
            state = fixture(self)
            original = state["client"]["api"]

            def guarded(method, path, data=None, _original=original, _scenario=scenario):
                result = _original(method, path, data)
                body = result["body"]
                if _scenario == "digest" and isinstance(body, list):
                    del body[0]["digest"]
                if _scenario == "published" and path.endswith("/releases/123"):
                    body["draft"] = False
                return result

            state["client"]["api"] = guarded
            with self.assertRaisesRegex(ToolError, "retained after failure"):
                publish_draft(state["options"], **state["deps"])
            if scenario == "published":
                self.assertEqual(state["uploads"], [])

    def test_github_status_parser_fails_closed_and_credentials_only_reach_github_child_environment(self):
        parsed = parse_api_response(ProcResult(1, None, "HTTP/2.0 404 Not Found\r\nContent-Type: application/json"
                                                      "\r\n\r\n{\"message\":\"Not Found\"}", ""))
        self.assertEqual(parsed["status"], 404)
        with self.assertRaisesRegex(ToolError, "lacks an HTTP status"):
            parse_api_response(ProcResult(1, None, "auth error token=private", ""))
        with self.assertRaisesRegex(ToolError, "invalid GitHub JSON"):
            parse_api_response(ProcResult(0, None, "HTTP/2.0 200 OK\n\nnot json", ""))
        source = {"PATH": "/bin", "HOME": "/home/test", "GH_TOKEN": "fake-auth", "OPENAI_API_KEY": "fake-provider",
                  "GH_DEBUG": "api", "GH_HOST": "evil.example", "UV_INDEX": "private"}
        self.assertNotIn("GH_TOKEN", safe_env(source))
        gh = safe_env(source, github=True)
        self.assertEqual(gh["GH_TOKEN"], "fake-auth")
        self.assertEqual(gh["GH_HOST"], "github.com")
        for key in ["OPENAI_API_KEY", "GH_DEBUG", "UV_INDEX"]:
            self.assertNotIn(key, gh)

    def test_gh_adapter_uses_stdin_for_release_json_and_uploads_by_id_without_clobber_or_credentials_in_argv(self):
        calls = []

        def fake_run(command, args, **kwargs):
            calls.append({"command": command, "args": args, "input": kwargs.get("input")})
            return ProcResult(0, None, "HTTP/2.0 201 Created\n\n{\"id\":123}", "")

        client = github_client(run=fake_run)
        client["api"]("POST", "repos/example/models/releases", {"draft": True})
        client["upload"]("example/models", 123, "/tmp/model_256_1.onnx")
        self.assertEqual(calls[0]["input"], "{\"draft\":true}")
        url = "https://uploads.github.com/repos/example/models/releases/123/assets?name=model_256_1.onnx"
        self.assertIn(url, calls[1]["args"])
        self.assertFalse(any("--clobber" in call["args"] for call in calls))

    def test_successful_parent_cannot_leave_an_owned_detached_stdio_descendant_running(self):
        code = ("import subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],\n"
                "                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,\n"
                "                         stderr=subprocess.DEVNULL)\n"
                "print(child.pid, flush=True)\n")
        result = run_process(sys.executable, ["-c", code], timeout_ms=5000)
        self.assertEqual(result.code, 0)
        pid = int(result.stdout.strip())
        self.addCleanup(lambda: _reap(pid))
        alive = True
        for _ in range(40):
            if not alive:
                break
            try:
                os.kill(pid, 0)
                import time
                time.sleep(0.025)
            except ProcessLookupError:
                alive = False
        self.assertFalse(alive, "descendant survived the process group kill")

    def test_bounded_subprocess_reports_timeout_output_overflow_without_leaking_child_stderr(self):
        with self.assertRaisesRegex(ToolError, "timed out") as caught:
            run_process(sys.executable, ["-c", "import sys, time; sys.stderr.write('fake-token'); "
                                              "sys.stderr.flush(); time.sleep(30)"], timeout_ms=50)
        self.assertNotIn("fake-token", str(caught.exception))
        with self.assertRaisesRegex(ToolError, "output limit"):
            run_process(sys.executable, ["-c", "import sys, time; sys.stdout.write('x' * 50000); "
                                              "sys.stdout.flush(); time.sleep(30)"], max_bytes=100, timeout_ms=2000)


def _reap(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass


class CliTest(unittest.TestCase):
    CLI = TOOL / "cli.py"

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(self.CLI), *args], capture_output=True, text=True, timeout=60)

    def test_help_and_unknown_or_duplicate_options_fail_without_touching_models(self):
        helped = self.run_cli("--help")
        self.assertEqual(helped.returncode, 0)
        self.assertIn("DTLN-aec ONNX artifact maintenance", helped.stdout)
        for args in [["frobnicate", "--variant", "256", "--version", "1.0.0"],
                     ["build", "--variant", "256", "--variant", "256", "--version", "1.0.0"],
                     ["build", "--variant", "256", "--version", "1.0.0", "--directory", "/tmp/x"],
                     ["verify", "--variant", "256", "--version", "1.0.0"],
                     ["publish", "--variant", "256", "--version", "1.0.0", "--directory", "/tmp/x",
                      "--repo", "o/r", "--target", TARGET]]:
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 1, args)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
