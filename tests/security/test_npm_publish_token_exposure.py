# Copyright 2026 Palantir Technologies, Inc.
#
# These tests are the spec for the docs-snippets npm release job. They fail while
# a dependency install or the TypeScript build in that job can read NPM_TOKEN.
# Run:
#   python3 tests/security/test_npm_publish_token_exposure.py

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_TOKEN = "placeholder-not-a-real-token"
TOKEN_NAME = "NPM_TOKEN"

# cimg/node Dockerfiles install these Node releases, and nodejs.org/dist/index.json
# records the npm bundled with each release. npm-publish uses cimg/node:22.14.
NODE_IMAGE_NPM = {
    "18.20": "10.8.2",
    "20.19": "10.8.2",
    "22.14": "10.9.2",
}


class Registry(http.server.ThreadingHTTPServer):
    def __init__(self, packuments, tarballs):
        self.packuments = packuments
        self.tarballs = tarballs
        super().__init__(("127.0.0.1", 0), RegistryHandler)

    @property
    def url(self):
        host, port = self.server_address[:2]
        return "http://%s:%s" % (host, port)


class RegistryHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urllib.parse.unquote(self.path.split("?", 1)[0])
        registry = self.server
        if path.startswith("/tarballs/"):
            blob = registry.tarballs.get(path[len("/tarballs/") :])
            if blob is None:
                self.send_error(404)
                return
            self._bytes(blob, "application/octet-stream")
            return
        name = path.lstrip("/")
        if name in registry.packuments:
            self._bytes(json.dumps(registry.packuments[name]).encode(), "application/json")
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self._bytes(b"{}", "application/json")

    def _bytes(self, payload, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def npm_cli(version):
    destination = Path(tempfile.gettempdir()) / "npm-under-test" / version
    binary = destination / "node_modules" / ".bin" / "npm"
    if not binary.exists():
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        subprocess.check_call(
            [
                "npm",
                "install",
                "npm@%s" % version,
                "--prefix",
                str(destination),
                "--ignore-scripts",
                "--no-audit",
                "--no-fund",
            ],
            stdout=subprocess.DEVNULL,
        )
    return binary


def semver_plan(npm_binary, spec):
    semver_path = npm_binary.resolve().parents[1] / "node_modules" / "semver"
    script = r"""
const semver = require(process.argv[1]);
const spec = process.argv[2];
const exact = semver.valid(spec);
if (exact) {
  const clean = semver.clean(spec);
  console.log(JSON.stringify({
    minimum: clean,
    attacker: null,
    outside: semver.inc(clean, "major"),
  }));
  process.exit(0);
}
const minimum = semver.minVersion(spec).version;
const candidates = [];
let cursor = minimum;
for (let i = 0; i < 12; i++) {
  cursor = semver.inc(cursor, "patch");
  if (cursor && semver.satisfies(cursor, spec)) candidates.push(cursor);
}
for (const release of ["minor", "major"]) {
  const bumped = semver.inc(minimum, release);
  if (bumped && semver.satisfies(bumped, spec)) candidates.push(bumped);
}
candidates.sort(semver.rcompare);
let outside = null;
for (const release of ["major", "minor", "patch"]) {
  const bumped = semver.inc(minimum, release);
  if (bumped && !semver.satisfies(bumped, spec)) {
    outside = bumped;
    break;
  }
}
console.log(JSON.stringify({
  minimum,
  attacker: candidates[0] || null,
  outside,
}));
"""
    completed = subprocess.run(
        ["node", "-e", script, str(semver_path), spec],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def package_tarball(name, version, role):
    # role is "benign", "attacker", or "outside".
    # Attacker and outside packages record whether NPM_TOKEN was visible.
    # They never write the value.
    root = Path(tempfile.mkdtemp())
    package_dir = root / "package"
    package_dir.mkdir()
    manifest = {
        "name": name,
        "version": version,
        "license": "Apache-2.0",
    }
    recorder = (
        "const fs = require('fs');\n"
        "const visible = Boolean(process.env.NPM_TOKEN);\n"
        "fs.appendFileSync(process.env.MARKER_PATH, "
        "['%s', process.env.npm_package_name + '@' + process.env.npm_package_version, "
        "visible ? 'token-visible' : 'token-absent', 'role=%s'].join(' ') + '\\n');\n"
    )
    if role != "benign":
        (package_dir / "install.js").write_text(recorder % ("install", role))
        manifest["scripts"] = {"install": "node install.js"}
    if name == "typescript":
        (package_dir / "tsc.js").write_text("#!/usr/bin/env node\n" + recorder % ("bin", role))
        os.chmod(package_dir / "tsc.js", 0o755)
        manifest["bin"] = {"tsc": "./tsc.js"}
    else:
        (package_dir / "index.d.ts").write_text(
            "export type SdkSnippets<T> = T;\nexport type PLATFORM_API_DOCS_SPEC = {};\n"
        )
    (package_dir / "package.json").write_text(json.dumps(manifest))
    tarball_path = root / "package.tgz"
    with tarfile.open(tarball_path, "w:gz") as archive:
        archive.add(package_dir, arcname="package")
    payload = tarball_path.read_bytes()
    shutil.rmtree(root)
    return payload


def build_registry(npm_binary, package_json, lockfile):
    specs = {}
    for field in ("dependencies", "devDependencies", "optionalDependencies"):
        specs.update(package_json.get(field) or {})
    locked = locked_versions(lockfile)
    packuments = {}
    tarballs = {}
    plans = {}
    for name, spec in specs.items():
        plan = semver_plan(npm_binary, spec)
        if name in locked:
            plan["locked"] = locked[name]
        plans[name] = plan
        versions = {}
        roles = {}
        roles[plan["minimum"]] = "benign"
        if plan.get("locked"):
            roles[plan["locked"]] = "benign"
        if plan["attacker"]:
            roles[plan["attacker"]] = "attacker"
        if plan["outside"]:
            roles[plan["outside"]] = "outside"
        for version, role in roles.items():
            payload = package_tarball(name, version, role)
            filename = "%s-%s.tgz" % (name.split("/")[-1], version)
            tarballs[filename] = payload
            integrity = (
                "sha512-"
                + __import__("base64").b64encode(hashlib.sha512(payload).digest()).decode()
            )
            versions[version] = {
                "name": name,
                "version": version,
                "dist": {
                    "shasum": hashlib.sha1(payload).hexdigest(),
                    "integrity": integrity,
                    "tarball": "http://placeholder/tarballs/" + filename,
                },
            }
            if role != "benign":
                # The registry manifest carries the install script. Arborist runs it
                # when hasInstallScript is set or scripts.install is present.
                versions[version]["hasInstallScript"] = True
                versions[version]["scripts"] = {"install": "node install.js"}
            if name == "typescript":
                versions[version]["bin"] = {"tsc": "./tsc.js"}
        outside = plan["outside"]
        packuments[name] = {
            "name": name,
            "dist-tags": {"latest": outside or plan["attacker"] or plan["minimum"]},
            "versions": versions,
        }
    return plans, packuments, tarballs


def locked_versions(lockfile):
    found = {}
    if not lockfile:
        return found

    def walk(node):
        if isinstance(node, dict):
            name = node.get("name")
            version = node.get("version")
            if (
                isinstance(name, str)
                and isinstance(version, str)
                and name
                in {
                    "typescript",
                    "@osdk/docs-spec-core",
                    "@osdk/docs-spec-platform",
                }
            ):
                found[name] = version
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(lockfile)
    return found


def substitute(text, parameters):
    import re

    def replace(match):
        key = match.group(1).strip()
        if key.startswith("parameters."):
            return str(parameters[key.split(".", 1)[1]])
        raise AssertionError("unsubstituted config expression: %s" % key)

    return re.sub(r"<<\s*([^>]+?)\s*>>", replace, text)


def expand_steps(config, steps, parameters):
    expanded = []
    for step in steps:
        if isinstance(step, str):
            if step == "checkout":
                expanded.append({"checkout": True})
                continue
            command = config["commands"][step]
            expanded.extend(expand_steps(config, command["steps"], parameters))
            continue
        if not isinstance(step, dict) or len(step) != 1:
            raise AssertionError("unsupported step %r" % (step,))
        key, value = next(iter(step.items()))
        if key == "checkout":
            expanded.append({"checkout": True})
        elif key == "run":
            if isinstance(value, str):
                command = substitute(value, parameters)
                environment = {}
            else:
                command = substitute(value["command"], parameters)
                environment = value.get("environment") or {}
            expanded.append({"run": command, "environment": environment})
        elif key in config.get("commands", {}):
            expanded.extend(expand_steps(config, config["commands"][key]["steps"], parameters))
        else:
            raise AssertionError("unsupported step %s" % key)
    return expanded


def workflow_invocations(config):
    invocations = []
    for workflow in config["workflows"].values():
        if not isinstance(workflow, dict):
            continue
        for item in workflow.get("jobs", []):
            if isinstance(item, str):
                invocations.append({"job": item, "parameters": {}, "context": None})
                continue
            job_name, body = next(iter(item.items()))
            body = body or {}
            contexts = body.get("context")
            matrix = (body.get("matrix") or {}).get("parameters") or {}
            if not matrix:
                invocations.append(
                    {
                        "job": job_name,
                        "parameters": {},
                        "context": contexts,
                    }
                )
                continue
            keys = list(matrix)
            values = [matrix[key] for key in keys]
            for combo in cartesian(values):
                parameters = dict(zip(keys, combo))
                invocations.append(
                    {
                        "job": job_name,
                        "parameters": parameters,
                        "context": contexts,
                    }
                )
    return invocations


def cartesian(lists):
    product = [()]
    for values in lists:
        product = [prefix + (value,) for prefix in product for value in values]
    return product


def command_references_token(command):
    return "$NPM_TOKEN" in command or "${NPM_TOKEN}" in command


def invocation_requests_token(config, invocation):
    job = config["jobs"][invocation["job"]]
    if TOKEN_NAME in (job.get("environment") or {}):
        return True
    for step in expand_steps(config, job["steps"], invocation["parameters"]):
        if step.get("run") and command_references_token(step["run"]):
            return True
        if TOKEN_NAME in (step.get("environment") or {}):
            return True
    return False


def context_names(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def project_token_is_global(config, invocations):
    """Project settings variables are injected into every job.

    A context is attached to a whole job. The config only receives NPM_TOKEN from
    outside the file. If any job uses the variable and attaches no context, the
    variable has to be a project variable and is present for every job. If every
    use is on a job that attaches a context, only jobs that attach one of those
    contexts receive it.
    """
    users = [item for item in invocations if invocation_requests_token(config, item)]
    if not users:
        return False, set()
    if any(not context_names(item["context"]) for item in users):
        return True, set()
    allowed = set()
    for item in users:
        allowed.update(context_names(item["context"]))
    return False, allowed


def job_has_project_token(config, invocation, invocations):
    global_token, contexts = project_token_is_global(config, invocations)
    if TOKEN_NAME in (config["jobs"][invocation["job"]].get("environment") or {}):
        return True
    if global_token:
        return True
    return bool(set(context_names(invocation["context"])) & contexts)


def prepare_workspace():
    workspace = Path(tempfile.mkdtemp(prefix="docs-snippets-release-"))
    shutil.copytree(
        REPO_ROOT / "docs-snippets-npm",
        workspace / "docs-snippets-npm",
        ignore=shutil.ignore_patterns("node_modules", "dist"),
    )
    shutil.copytree(REPO_ROOT / "scripts", workspace / "scripts")
    subprocess.check_call(["git", "init"], cwd=workspace, stdout=subprocess.DEVNULL)
    subprocess.check_call(
        ["git", "config", "user.email", "release-test@example.com"], cwd=workspace
    )
    subprocess.check_call(["git", "config", "user.name", "release-test"], cwd=workspace)
    subprocess.check_call(["git", "add", "."], cwd=workspace, stdout=subprocess.DEVNULL)
    subprocess.check_call(
        ["git", "commit", "-m", "workspace"], cwd=workspace, stdout=subprocess.DEVNULL
    )
    subprocess.check_call(["git", "tag", "v0.0.0-test"], cwd=workspace)
    return workspace


def run_job(npm_binary, invocation, has_token):
    config_path = REPO_ROOT / ".circleci" / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    invocations = workflow_invocations(config)
    # Recompute from the same config the job runner uses.
    assert has_token == job_has_project_token(config, invocation, invocations)
    workspace = prepare_workspace()
    home = Path(tempfile.mkdtemp(prefix="npm-home-"))
    marker = home / "marker.txt"
    marker.write_text("")
    package_json = json.loads((workspace / "docs-snippets-npm" / "package.json").read_text())
    lock_path = workspace / "docs-snippets-npm" / "package-lock.json"
    lockfile = json.loads(lock_path.read_text()) if lock_path.exists() else None
    plans, packuments, tarballs = build_registry(npm_binary, package_json, lockfile)
    registry = Registry(packuments, tarballs)
    thread = threading.Thread(target=registry.serve_forever, daemon=True)
    thread.start()
    try:
        base = registry.url
        for manifest in packuments.values():
            for version in manifest["versions"].values():
                version["dist"]["tarball"] = version["dist"]["tarball"].replace(
                    "http://placeholder", base
                )
        job = config["jobs"][invocation["job"]]
        steps = expand_steps(config, job["steps"], invocation["parameters"])
        job_env = {
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "TERM": "dumb",
            "MARKER_PATH": str(marker),
            "npm_config_registry": base + "/",
            "npm_config_cache": str(home / "cache"),
        }
        if has_token:
            job_env[TOKEN_NAME] = PLACEHOLDER_TOKEN
        results = []
        for step in steps:
            if step.get("checkout"):
                continue
            command = step["run"]
            if "npm publish" in command:
                # Publishing would upload a package. The install and build steps
                # already ran in earlier steps of this job.
                continue
            step_env = dict(job_env)
            step_env.update({key: str(value) for key, value in step["environment"].items()})
            # The shell command is `npm`, so the version under test has to be first on PATH.
            step_env["PATH"] = str(npm_binary.parent) + os.pathsep + step_env["PATH"]
            # CircleCI runs `/bin/bash -eo pipefail` for each step.
            completed = subprocess.run(
                ["bash", "-eo", "pipefail", "-c", command],
                cwd=workspace,
                env=step_env,
                capture_output=True,
                text=True,
            )
            results.append(completed)
            if completed.returncode != 0:
                break
        logs = sorted((home / "cache" / "_logs").glob("*-debug-0.log"))
        log_text = logs[-1].read_text() if logs else ""
        error_lines = [
            line
            for line in log_text.splitlines()
            if " error " in line
            or line.startswith("npm error")
            or "ERR" in line
            or "integrity" in line.lower()
        ]
        return {
            "plans": plans,
            "marker": marker.read_text(),
            "results": results,
            "log": "\n".join(error_lines) or log_text[-4000:],
        }
    finally:
        registry.shutdown()
        registry.server_close()
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


def release_cases(config):
    cases = []
    invocations = workflow_invocations(config)
    for invocation in invocations:
        job = config["jobs"][invocation["job"]]
        image = ""
        docker = job.get("docker") or []
        if docker:
            image = substitute(docker[0].get("image", ""), invocation["parameters"])
        if not image.startswith("cimg/node:"):
            continue
        node_version = image.split(":", 1)[1]
        steps = expand_steps(config, job["steps"], invocation["parameters"])
        if not any(step.get("run") and "npm install" in step["run"] for step in steps):
            continue
        cases.append(
            {
                "job": invocation["job"],
                "node": node_version,
                "npm": NODE_IMAGE_NPM[node_version],
                "invocation": invocation,
                "has_token": job_has_project_token(config, invocation, invocations),
            }
        )
    return cases


class NpmPublishTokenExposureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load((REPO_ROOT / ".circleci" / "config.yml").read_text())
        cls.cases = release_cases(config)
        if not cls.cases:
            raise AssertionError("no npm install jobs found in the CircleCI workflow")
        for version in sorted({case["npm"] for case in cls.cases}):
            npm_cli(version)

    def test_dependency_lifecycle_and_build_cannot_read_npm_token(self):
        self.assertGreaterEqual(len(self.cases), 7)
        for case in self.cases:
            with self.subTest(job=case["job"], node=case["node"], npm=case["npm"]):
                outcome = run_job(npm_cli(case["npm"]), case["invocation"], case["has_token"])
                failed = [result for result in outcome["results"] if result.returncode != 0]
                if failed:
                    result = failed[0]
                    self.fail(
                        "release step exited %s\nstdout:\n%s\nstderr:\n%s\nlog:\n%s"
                        % (result.returncode, result.stdout, result.stderr, outcome["log"])
                    )
                marker = outcome["marker"]
                for line in marker.splitlines():
                    self.assertNotIn(
                        "role=outside",
                        line,
                        "a version outside the declared range ran: %s" % marker,
                    )
                attacker_lines = [
                    line
                    for line in marker.splitlines()
                    if "role=attacker" in line and "token-visible" in line
                ]
                self.assertEqual(
                    attacker_lines,
                    [],
                    "dependency code selected from a floating range read %s during %s "
                    "(node %s, npm %s).\nmarker:\n%s\nplans:\n%s"
                    % (
                        TOKEN_NAME,
                        case["job"],
                        case["node"],
                        case["npm"],
                        marker,
                        json.dumps(outcome["plans"], indent=2),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
