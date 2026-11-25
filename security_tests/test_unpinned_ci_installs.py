"""Failing tests for unpinned CircleCI installs.

These tests execute the install commands in ``.circleci/config.yml``, ``tox.ini``,
and ``pyproject.toml`` against a local index that publishes a newer release of
each unpinned or open-ranged package. Each release's build backend records
that it ran. The tests fail while that code runs and pass once those installs
refuse every release except an exact pinned or locked one.

The index stands in for PyPI. Pip and Poetry are the real installers.

Not covered here: ``docs-snippets-npm`` runs ``npm install`` with no lockfile.
That is a separate installer from the PyPI path these tests execute.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
import textwrap
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / ".circleci" / "config.yml"
HARNESS_PACKAGES = ("poetry==2.5.2", "tox==4.67.16")
PASSWORD_NAME = "PYPI_PASSWORD"


def _job_commands(config: dict, job_name: str) -> list[str]:
    commands = []
    for step in config["jobs"][job_name]["steps"]:
        if not isinstance(step, dict) or "run" not in step:
            continue
        run = step["run"]
        commands.append(run if isinstance(run, str) else run["command"])
    return commands


def _matrix_cell(config: dict) -> tuple[str, str, str]:
    """A python/pydantic/httpx cell the test workflow actually runs."""
    for job in config["workflows"]["build"]["jobs"]:
        if not isinstance(job, dict) or "test" not in job:
            continue
        matrix = job["test"].get("matrix", {}).get("parameters", {})
        versions = matrix.get("python_version") or []
        if "3.12" not in versions:
            continue
        return "3.12", matrix["pydantic_version"][0], matrix["httpx_version"][0]
    raise AssertionError("config has no python 3.12 test matrix cell")


class _Index(threading.Thread):
    def __init__(self, root: Path) -> None:
        super().__init__(daemon=True)
        self.root = root
        self.packages = root / "packages"
        self.simple = root / "simple"
        self.packages.mkdir(parents=True)
        self.simple.mkdir(parents=True)
        self.requests = 0
        handler = self._handler()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._httpd.server_address[1]

    @property
    def simple_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/simple"

    def _handler(self):
        index = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                index.requests += 1
                rel = self.path.split("?", 1)[0].lstrip("/")
                target = None
                if rel.startswith("simple/"):
                    target = index.simple / rel[len("simple/") :].strip("/") / "index.html"
                    content_type = "text/html"
                elif rel.startswith("packages/"):
                    target = index.packages / rel[len("packages/") :]
                    content_type = "application/octet-stream"
                if target is None or not target.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler

    def run(self) -> None:
        self._httpd.serve_forever()

    def add(self, page_name: str, filename: str, data: bytes) -> None:
        (self.packages / filename).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        page = self.simple / page_name
        page.mkdir(exist_ok=True)
        index = page / "index.html"
        line = f'<a href="../../packages/{filename}#sha256={digest}">{filename}</a>\n'
        if index.exists():
            index.write_text(index.read_text().replace("</body></html>", line + "</body></html>"))
        else:
            index.write_text("<!DOCTYPE html><html><body>\n" + line + "</body></html>\n")

    def add_payload_sdist(self, name: str, version: str, sentinel: Path) -> None:
        folder = f"{name}-{version}"
        src = self.root / "src" / folder
        if src.exists():
            shutil.rmtree(src)
        src.mkdir(parents=True)
        dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
        backend = textwrap.dedent(
            f"""
            import os
            from pathlib import Path

            def _mark():
                destination = Path({str(sentinel)!r})
                destination.mkdir(parents=True, exist_ok=True)
                channel = "env" if {PASSWORD_NAME!r} in os.environ else ""
                if not channel:
                    proc = Path("/proc")
                    if proc.is_dir():
                        for entry in proc.iterdir():
                            if not entry.name.isdigit():
                                continue
                            try:
                                raw = (entry / "environ").read_bytes()
                            except OSError:
                                continue
                            if b"{PASSWORD_NAME}=" in raw:
                                channel = "proc"
                                break
                (destination / {name!r}).write_text(channel or "ran")

            _mark()

            def get_requires_for_build_wheel(config_settings=None):
                return []

            def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
                meta = Path(metadata_directory) / {dist_info!r}
                meta.mkdir(parents=True, exist_ok=True)
                (meta / "METADATA").write_text(
                    "Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\n"
                )
                (meta / "WHEEL").write_text(
                    "Wheel-Version: 1.0\\nGenerator: security-test\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n"
                )
                return meta.name

            def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
                import base64
                import hashlib
                import zipfile

                whl_name = "{name.replace("-", "_")}-{version}-py3-none-any.whl"
                whl_path = Path(wheel_directory) / whl_name
                info = {dist_info!r}
                files = {{
                    info + "/METADATA": "Metadata-Version: 2.1\\nName: {name}\\nVersion: {version}\\n",
                    info + "/WHEEL": "Wheel-Version: 1.0\\nGenerator: security-test\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
                }}
                record = []
                with zipfile.ZipFile(whl_path, "w") as archive:
                    for arcname, content in files.items():
                        data = content.encode()
                        archive.writestr(arcname, data)
                        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                        record.append(f"{{arcname}},sha256={{digest}},{{len(data)}}")
                    record.append(info + "/RECORD,,")
                    archive.writestr(info + "/RECORD", "\\n".join(record) + "\\n")
                return whl_name
            """
        )
        (src / "payload_backend.py").write_text(backend)
        (src / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [build-system]
                requires = []
                build-backend = "payload_backend"
                backend-path = ["."]

                [project]
                name = "{name}"
                version = "{version}"
                """
            ).lstrip()
        )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(src, arcname=folder)
        self.add(name, f"{folder}.tar.gz", buffer.getvalue())

    def add_inert_wheel(self, name: str, version: str) -> None:
        self._add_wheel(name, version, {})

    def _add_wheel(self, name: str, version: str, modules: dict[str, str]) -> None:
        info = f"{name.replace('-', '_')}-{version}.dist-info"
        files = {
            f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: security-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        }
        files.update(modules)
        buffer = io.BytesIO()
        record = []
        with zipfile.ZipFile(buffer, "w") as archive:
            for arcname, content in files.items():
                data = content.encode()
                archive.writestr(arcname, data)
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                record.append(f"{arcname},sha256={digest},{len(data)}")
            record.append(f"{info}/RECORD,,")
            archive.writestr(f"{info}/RECORD", "\n".join(record) + "\n")
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        self.add(name, filename, buffer.getvalue())


def _executed(sentinel: Path) -> dict[str, str]:
    if not sentinel.exists():
        return {}
    return {path.name: path.read_text() for path in sorted(sentinel.iterdir())}


def _assert_not_executed(sentinel: Path, output: str, requests: int) -> None:
    executed = _executed(sentinel)
    assert requests > 0, f"installer never contacted the index\n{output[-2000:]}"
    assert not executed, "a release from the extra index ran code during install:\n" + "\n".join(
        f"{name}: {detail}" for name, detail in executed.items()
    )


@pytest.fixture(scope="session")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("harness")
    subprocess.check_call(
        [os.sys.executable, "-m", "pip", "install", "--target", str(target), *HARNESS_PACKAGES],
    )
    return target


@pytest.fixture()
def index(tmp_path: Path) -> _Index:
    server = _Index(tmp_path / "index")
    server.add_inert_wheel("setuptools", "68.1.2")
    server.add_inert_wheel("wheel", "0.42.0")
    server.start()
    yield server
    server._httpd.shutdown()


@pytest.fixture()
def sentinel(tmp_path: Path) -> Path:
    path = tmp_path / "sentinel"
    path.mkdir()
    return path


def _child_env(sentinel: Path, index: _Index, harness: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env[PASSWORD_NAME] = "placeholder-not-a-secret"
    env["PYPI_USERNAME"] = "placeholder-user"
    env["PIP_INDEX_URL"] = index.simple_url
    env["PIP_TRUSTED_HOST"] = "127.0.0.1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    home = sentinel.parent / "home"
    home.mkdir(exist_ok=True)
    env["HOME"] = str(home)
    env["PYTHONUSERBASE"] = str(sentinel.parent / "userbase")
    if harness is not None:
        env["PYTHONPATH"] = str(harness) + os.pathsep + env.get("PYTHONPATH", "")
        env["PATH"] = str(harness / "bin") + os.pathsep + env["PATH"]
        env["POETRY_VIRTUALENVS_IN_PROJECT"] = "true"
        env["POETRY_CACHE_DIR"] = str(sentinel.parent / "poetry-cache")
    return env


def _publish_payload_releases(index: _Index, sentinel: Path) -> None:
    """Versions that satisfy the ranges in pyproject.toml and tox.ini."""
    releases = {
        "annotated-types": ("0.99.0",),
        "pydantic": ("2.99.0", "2.12.99", "2.11.99", "2.10.99", "2.9.99", "2.8.99", "2.7.99", "2.6.99"),
        "httpx": ("0.99.0", "0.28.99", "0.27.99", "0.26.99", "0.25.99"),
        "httpx-sse": ("0.99.0",),
        "typing-extensions": ("4.99.0",),
        "h11": ("0.99.0",),
        "retrying": ("1.99.0",),
        "pyyaml": ("6.99.0",),
        "expects": ("99.0.0",),
        "mockito": ("99.0.0",),
        "pytest": ("99.0.0",),
        "pytest-asyncio": ("99.0.0",),
        "uvicorn": ("99.0.0",),
        "fastapi": ("99.0.0",),
        "pyright": ("99.0.0",),
        "click": ("99.0.0",),
        "polars": ("99.0.0",),
        "duckdb": ("99.0.0",),
        "pyarrow": ("99.0.0",),
        "tox": ("99.0.0",),
        "black": ("99.0.0",),
    }
    for name, versions in releases.items():
        for version in versions:
            index.add_payload_sdist(name, version, sentinel)


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_circleci_pip_install_does_not_run_a_newer_tox(index: _Index, sentinel: Path) -> None:
    config = _config()
    commands = [command for job in ("test", "black") for command in _job_commands(config, job)]
    pip_commands = [command.strip() for command in commands if command.strip().startswith("pip install")]
    assert pip_commands, "test and black jobs no longer install with pip"
    index.add_payload_sdist("tox", "99.0.0", sentinel)
    env = _child_env(sentinel, index)
    for command in pip_commands:
        subprocess.run(command, shell=True, env=env, text=True, capture_output=True, check=False)
    _assert_not_executed(sentinel, "\n".join(pip_commands), index.requests)


def test_tox_deps_do_not_run_newer_releases(index: _Index, sentinel: Path, harness: Path) -> None:
    config = _config()
    python_version, pydantic_version, httpx_version = _matrix_cell(config)
    _publish_payload_releases(index, sentinel)
    env = _child_env(sentinel, index, harness)
    env["PYDANTIC_VERSION"] = f"{pydantic_version}.*"
    env["HTTPX_VERSION"] = f"{httpx_version}.*"
    work = sentinel.parent / "toxwork"
    completed = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "tox",
            "-e",
            f"py{python_version}",
            "--recreate",
            "--notest",
            "--workdir",
            str(work),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    _assert_not_executed(sentinel, output, index.requests)


def test_pinned_black_dependency_does_not_run_a_newer_release(
    index: _Index, sentinel: Path, harness: Path
) -> None:
    index.add_payload_sdist("black", "99.0.0", sentinel)
    env = _child_env(sentinel, index, harness)
    work = sentinel.parent / "toxwork-black"
    completed = subprocess.run(
        [os.sys.executable, "-m", "tox", "-e", "black", "--recreate", "--notest", "--workdir", str(work)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    assert "black==24.1.1" in output or "black == 24.1.1" in output
    _assert_not_executed(sentinel, output, index.requests)


def _poetry_project(root: Path, index: _Index) -> Path:
    project = root / "project"
    shutil.copytree(
        REPO_ROOT,
        project,
        ignore=shutil.ignore_patterns(".git", ".venv", "security_tests", "__pycache__"),
        symlinks=True,
    )
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text()
        + textwrap.dedent(
            f"""

            [[tool.poetry.source]]
            name = "hostile"
            url = "{index.simple_url}"
            priority = "primary"
            """
        )
    )
    return project


def test_poetry_install_does_not_run_in_range_releases(
    index: _Index, sentinel: Path, harness: Path
) -> None:
    config = _config()
    install_commands = [
        command.strip()
        for command in _job_commands(config, "test")
        if "poetry" in command and "install" in command
    ]
    assert install_commands, "test job no longer installs with poetry"
    _publish_payload_releases(index, sentinel)
    project = _poetry_project(sentinel.parent, index)
    env = _child_env(sentinel, index, harness)
    output = []
    for command in install_commands:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=project,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        output.append(completed.stdout)
        output.append(completed.stderr)
    _assert_not_executed(sentinel, "\n".join(output), index.requests)


def test_publish_build_does_not_run_a_newer_build_backend(
    index: _Index, sentinel: Path, harness: Path
) -> None:
    config = _config()
    publish_commands = _job_commands(config, "publish")
    assert any("poetry publish" in command and "--build" in command for command in publish_commands)
    assert any(f"${PASSWORD_NAME}" in command for command in publish_commands)
    backend = textwrap.dedent(
        f"""
        import os
        from pathlib import Path

        destination = Path({str(sentinel)!r})
        destination.mkdir(parents=True, exist_ok=True)
        channel = "env" if {PASSWORD_NAME!r} in os.environ else ""
        if not channel:
            proc = Path("/proc")
            if proc.is_dir():
                for entry in proc.iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        raw = (entry / "environ").read_bytes()
                    except OSError:
                        continue
                    if b"{PASSWORD_NAME}=" in raw:
                        channel = "proc"
                        break
        (destination / "poetry-core").write_text(channel or "ran")

        def get_requires_for_build_wheel(config_settings=None):
            return []

        def get_requires_for_build_sdist(config_settings=None):
            return []

        def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
            raise RuntimeError("payload")

        def build_sdist(sdist_directory, config_settings=None):
            raise RuntimeError("payload")
        """
    )
    index._add_wheel(
        "poetry-core",
        "99.0.0",
        {
            "poetry/__init__.py": "",
            "poetry/core/__init__.py": "",
            "poetry/core/masonry/__init__.py": "",
            "poetry/core/masonry/api.py": backend,
        },
    )
    project = _poetry_project(sentinel.parent, index)
    env = _child_env(sentinel, index, harness)
    # poetry publish --build calls the build command before any upload.
    completed = subprocess.run(
        [os.sys.executable, "-m", "poetry", "build"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    _assert_not_executed(sentinel, output, index.requests)
