"""One Python dependency bundle for the guest, built once and delivered once.

WHY THIS MODULE EXISTS.  The guest needs the same handful of Python packages in
two roles that used to be served by two independent mechanisms:

* task **setup** generates the .xlsx/.docx/.pptx fixtures a task starts from,
  and runs inside ``env.reset()``;
* the **agent's** ``bash`` tool reads and writes those same files afterwards.

The two happen in that order with nothing in between, so whatever serves setup
also serves the agent -- but only if it lands before ``reset()`` returns.  The
provider's ``_wait_for_vm_ready()`` is the one hook that fits, because every
``reset()`` destroys the container and rebuilds it from the read-only qcow2:
anything installed earlier is already gone by the time setup runs.

The two mechanisms it replaces disagreed in ways that were invisible from
inside the guest.  One ``pip install --user``-ed three pinned packages from
PyPI, which needs the guest to have working network; the other shipped six
packages copied out of the outer container, which does not.  ``~/.local``
precedes the bundle's directory on ``sys.path``, so where they overlapped the
pip copy won silently -- python-docx resolved to 1.1.2 while the bundle sitting
in /opt said 1.2.0, and the import check at the end of bootstrap could not tell
which of the two it had just verified.

WHY COPYING COMPILED EXTENSIONS IS LEGAL HERE.  Both the outer container and
the guest run CPython 3.10 (3.10.19 and 3.10.x respectively), so ``.so`` files
tagged ``cpython-310-x86_64-linux-gnu`` load on both sides.  This is what lets
pandas and OpenCV travel in a tarball at all.  ``lxml`` is still deliberately
excluded: the guest's own desktop server already depends on its distro build,
and replacing that with the host's is how you break the server rather than the
task.

WHY ``distribution.files`` AND NOT THE MODULE DIRECTORY.  OpenCV's extension
modules load their shared libraries through an RPATH of
``$ORIGIN/../opencv_python_headless.libs`` -- a *sibling* of the ``cv2``
package.  Copying only ``cv2/`` produces a package that imports on the host and
dies in the guest.  Reading the distribution's own file list instead picks up
every top-level entry it owns, siblings included, without a special case.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import shlex
import stat
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import requests
from filelock import FileLock

logger = logging.getLogger("desktopenv.pydeps_bundle")

# Bump when the contents or the layout change, so a guest holding an older
# bundle is refreshed rather than trusted.
PYDEPS_BUNDLE_VERSION = 5
PYDEPS_MANIFEST = "mage_hybrid_pydeps.json"
PYDEPS_READY_VERSION = 1
GUEST_PYTHON_VERSION = (3, 10)

REMOTE_PYDEPS_DIR = "/opt/mage-hybrid-pydeps"
REMOTE_PYDEPS_MANIFEST = f"{REMOTE_PYDEPS_DIR}/{PYDEPS_MANIFEST}"
GUEST_SITE_DIR = "/usr/local/lib/python3.10/dist-packages"
GUEST_PYDEPS_PTH = f"{GUEST_SITE_DIR}/mage_hybrid_pydeps.pth"
REMOTE_PYDEPS_TARBALL = "/tmp/mage_hybrid_pydeps.tar.gz"
PYDEPS_BUILD_LOCK_TIMEOUT = int(
    os.environ.get("OSWORLD_PYDEPS_BUILD_LOCK_TIMEOUT", "600")
)

# (import name, distribution name).  Every entry is copied out of the outer
# container, so every entry must be importable there or the build fails loudly.
PYDEPS_DISTRIBUTIONS = (
    # Binary packages in this bundle must share one NumPy ABI. The guest image
    # has NumPy 1.21.5, which is too old for the bundled pandas and OpenCV.
    ("numpy", "numpy"),
    # Office file formats: what setup writes and what the agent edits.
    ("openpyxl", "openpyxl"),
    ("et_xmlfile", "et_xmlfile"),
    ("xlsxwriter", "XlsxWriter"),
    ("docx", "python-docx"),
    ("pptx", "python-pptx"),
    ("defusedxml", "defusedxml"),
    ("typing_extensions", "typing_extensions"),
    # PDF: the guest has the poppler CLI tools but no Python reader, and 1,013
    # tasks hit `No module named 'pypdf'` before this.
    ("pypdf", "pypdf"),
    # Tabular data. pandas needs all four of the following at import time.
    ("pandas", "pandas"),
    ("dateutil", "python-dateutil"),
    ("six", "six"),
    ("pytz", "pytz"),
    ("tzdata", "tzdata"),
    # Vision.  The GUI build, not `-headless`, because that is what is actually
    # installed: both dist-infos exist in the outer image and the later install
    # overwrote `cv2/`, so the on-disk extension carries
    # `RPATH $ORIGIN/../opencv_python.libs`.  Shipping the headless metadata's
    # sibling directory hands the guest a set of differently hash-suffixed
    # libraries and `import cv2` dies on a missing libpng.  _verify_owns_disk
    # below is what turns that class of mistake into a build error.
    ("cv2", "opencv-python"),
)

# lxml is verified, never shipped -- see the module docstring.
PYDEPS_VERIFY_IMPORTS = (
    "numpy",
    "openpyxl",
    "et_xmlfile",
    "xlsxwriter",
    "docx",
    "pptx",
    "defusedxml",
    "typing_extensions",
    "pypdf",
    "pandas",
    "dateutil",
    "six",
    "pytz",
    "tzdata",
    "cv2",
    "lxml.etree",
)


def _require_guest_compatible_python() -> None:
    current = sys.version_info[:2]
    if current != GUEST_PYTHON_VERSION:
        expected = ".".join(map(str, GUEST_PYTHON_VERSION))
        actual = ".".join(map(str, current))
        raise RuntimeError(
            "Hybrid Python dependency bundle requires host CPython "
            f"{expected} to match the VM ABI; current host is {actual}"
        )


def _distribution_top_level_names(distribution) -> set[str]:
    """Top-level entries in site-packages that this distribution owns.

    ``files`` also lists console scripts as ``../../../bin/x`` and stray
    ``__pycache__`` entries; neither belongs in the bundle.
    """
    names: set[str] = set()
    for relative in distribution.files or []:
        parts = PurePosixPath(str(relative)).parts
        if not parts or parts[0] in ("..", "__pycache__"):
            continue
        names.add(parts[0])
    return names


def _verify_owns_disk(distribution, distribution_name: str, base: Path) -> None:
    """Refuse a distribution whose RECORD disagrees with the files on disk.

    Two distributions can claim the same top-level package -- opencv-python and
    opencv-python-headless both own ``cv2`` -- and pip lets the second install
    overwrite the first's files while leaving both ``.dist-info`` directories in
    place.  Reading the loser's file list then produces a bundle that pairs one
    build's extension module with the other build's bundled shared libraries,
    and the guest fails at import on a library whose hash suffix does not match.
    RECORD carries a size per file, so the disagreement is cheap to detect.
    """
    matched = mismatched = 0
    offenders = []
    for relative in distribution.files or []:
        size = getattr(relative, "size", None)
        if size is None:
            continue
        path = base / str(relative)
        if not path.is_file():
            continue
        if path.stat().st_size == size:
            matched += 1
        else:
            mismatched += 1
            if len(offenders) < 3:
                offenders.append(str(relative))
    if mismatched and mismatched > matched // 4:
        raise RuntimeError(
            f"{distribution_name} is installed but its files on disk belong to "
            f"another distribution ({mismatched} of {matched + mismatched} sized "
            f"entries disagree, e.g. {offenders}). Bundling it would mix two "
            "builds; name the distribution that actually owns these files."
        )


def _portable_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Make bundle members readable by the unprivileged desktop user."""
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o755 if info.mode & 0o111 else 0o644
    return info


def _collect_entries() -> tuple[list[tuple[str, Path]], dict[str, str], set[str]]:
    _require_guest_compatible_python()
    archive_entries: list[tuple[str, Path]] = []
    distribution_versions: dict[str, str] = {}
    required_members: set[str] = {PYDEPS_MANIFEST}
    seen: set[str] = set()

    for import_name, distribution_name in PYDEPS_DISTRIBUTIONS:
        if importlib.util.find_spec(import_name) is None:
            raise RuntimeError(
                f"Host Python package required for VM bundle: {import_name} "
                f"(distribution {distribution_name})"
            )
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Host Python distribution metadata required for VM bundle: "
                f"{distribution_name}"
            ) from exc

        base = Path(str(distribution.locate_file(""))).resolve()
        _verify_owns_disk(distribution, distribution_name, base)
        top_level = _distribution_top_level_names(distribution)
        if not top_level:
            raise RuntimeError(
                f"Distribution reports no files for VM bundle: {distribution_name}"
            )
        for name in sorted(top_level):
            source = base / name
            if not source.exists():
                # A RECORD can outlive the file it names; ignore the ghost
                # rather than failing the whole bundle over it.
                continue
            if name in seen:
                continue
            seen.add(name)
            archive_entries.append((name, source))
            required_members.add(name)

        distribution_versions[distribution_name] = distribution.version
        # Pin the one entry that proves this distribution actually travelled.
        module_spec = importlib.util.find_spec(import_name)
        if module_spec is not None and module_spec.submodule_search_locations:
            required_members.add(f"{import_name}/__init__.py")

    return archive_entries, distribution_versions, required_members


def expected_manifest() -> dict:
    """What the guest's manifest must say for the bundle to be considered current."""
    _, distribution_versions, _ = _collect_entries()
    return {
        "bundle_version": PYDEPS_BUNDLE_VERSION,
        "distributions": distribution_versions,
    }


def _ready_sidecar_path(tarball: Path) -> Path:
    return tarball.with_name(f"{tarball.name}.ready.json")


def _regular_file_stat(path: Path):
    try:
        result = os.stat(path, follow_symlinks=False)
    except OSError:
        return None
    return result if stat.S_ISREG(result.st_mode) else None


def _read_ready_sidecar(sidecar: Path):
    """Read a regular sidecar without following a symlink where supported."""
    expected_stat = _regular_file_stat(sidecar)
    if expected_stat is None:
        return None

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(sidecar, flags)
        opened_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_dev != expected_stat.st_dev
            or opened_stat.st_ino != expected_stat.st_ino
        ):
            return None
        handle = os.fdopen(fd, "r", encoding="utf-8")
        fd = -1
        with handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def _tarball_identity(file_stat) -> dict[str, int]:
    return {
        "size": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "ctime_ns": file_stat.st_ctime_ns,
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
    }


def _ready_sidecar_matches(tarball: Path, manifest: dict) -> bool:
    """Check cache readiness without opening the comparatively large gzip."""
    ready = _read_ready_sidecar(_ready_sidecar_path(tarball))
    if not isinstance(ready, dict):
        return False
    tarball_stat = _regular_file_stat(tarball)
    return (
        tarball_stat is not None
        and tarball_stat.st_size > 0
        and ready.get("ready_version") == PYDEPS_READY_VERSION
        and ready.get("manifest") == manifest
        and ready.get("tarball") == tarball.name
        and all(
            ready.get(key) == value
            for key, value in _tarball_identity(tarball_stat).items()
        )
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_ready_sidecar(tarball: Path, manifest: dict) -> None:
    tarball_stat = _regular_file_stat(tarball)
    if tarball_stat is None:
        raise RuntimeError(f"Python dependency bundle is not a regular file: {tarball}")
    ready = {
        "ready_version": PYDEPS_READY_VERSION,
        "manifest": manifest,
        "tarball": tarball.name,
        **_tarball_identity(tarball_stat),
    }
    sidecar = _ready_sidecar_path(tarball)
    fd, temp_name = tempfile.mkstemp(prefix=sidecar.name, dir=sidecar.parent)
    temp_path = Path(temp_name)
    try:
        handle = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1
        with handle:
            json.dump(ready, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, sidecar)
        _fsync_directory(sidecar.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def _cached_tarball_is_valid(
    tarball: Path,
    manifest: dict,
    required_members: set[str],
) -> bool:
    try:
        with tarfile.open(tarball, "r:gz") as archive:
            cached_file = archive.extractfile(PYDEPS_MANIFEST)
            cached_manifest = json.load(cached_file) if cached_file else None
            cached_members = set(archive.getnames())
        return cached_manifest == manifest and required_members.issubset(cached_members)
    except (
        KeyError,
        EOFError,
        OSError,
        UnicodeDecodeError,
        tarfile.TarError,
        json.JSONDecodeError,
    ):
        return False


def ensure_python_deps_tarball() -> Path:
    """Build (or reuse) the guest dependency bundle and return its path.

    The cache key is a hash of the manifest rather than the version strings
    joined together: fourteen distributions spell a filename past the 255-byte
    limit, and the failure would only show up on whichever machine first added
    the fourteenth.
    """
    archive_entries, distribution_versions, required_members = _collect_entries()
    manifest = {
        "bundle_version": PYDEPS_BUNDLE_VERSION,
        "distributions": distribution_versions,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(manifest_bytes).hexdigest()[:16]

    cache_dir = Path(
        os.environ.get("MAGE_HYBRID_CACHE_DIR", "~/.cache/osworld-mage-hybrid")
    ).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball = cache_dir / f"pydeps-v{PYDEPS_BUNDLE_VERSION}-{digest}.tar.gz"
    if _ready_sidecar_matches(tarball, manifest):
        return tarball

    lock = FileLock(
        str(cache_dir / f"pydeps-bundle-{digest}.lock"),
        timeout=PYDEPS_BUILD_LOCK_TIMEOUT,
    )
    with lock:
        if _ready_sidecar_matches(tarball, manifest):
            return tarball

        if _regular_file_stat(tarball) is not None:
            if _cached_tarball_is_valid(tarball, manifest, required_members):
                _write_ready_sidecar(tarball, manifest)
                return tarball
            else:
                logger.warning(
                    "Rebuilding invalid cached Python dependency bundle: %s",
                    tarball,
                )

        fd, temp_name = tempfile.mkstemp(prefix=tarball.name, dir=cache_dir)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with tarfile.open(temp_path, "w:gz") as archive:
                for arcname, source_path in archive_entries:
                    archive.add(
                        source_path,
                        arcname=arcname,
                        filter=_portable_tar_info,
                    )
                info = tarfile.TarInfo(PYDEPS_MANIFEST)
                info.size = len(manifest_bytes)
                info.mtime = int(time.time())
                archive.addfile(info, io.BytesIO(manifest_bytes))
            with temp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_path, tarball)
            _fsync_directory(cache_dir)
            _write_ready_sidecar(tarball, manifest)
        finally:
            temp_path.unlink(missing_ok=True)
        logger.info(
            "Built guest Python dependency bundle %s (%.1f MB, %d distributions)",
            tarball.name,
            tarball.stat().st_size / 1e6,
            len(distribution_versions),
        )
        return tarball


def guest_extract_script(remote_tarball: str) -> str:
    """Shell to unpack the bundle in the guest.  Must run as root: /opt is root's.

    The ``.pth`` prepends the directory to ``sys.path``. The bundle is a coherent
    dependency set, so letting an older guest package win can mix incompatible
    NumPy, pandas and OpenCV binaries.
    """
    return (
        f"rm -rf {shlex.quote(REMOTE_PYDEPS_DIR)}; "
        f"mkdir -p {shlex.quote(REMOTE_PYDEPS_DIR)}; "
        f"tar xzf {shlex.quote(remote_tarball)} -C {shlex.quote(REMOTE_PYDEPS_DIR)}; "
        f"mkdir -p {shlex.quote(GUEST_SITE_DIR)}; "
        "printf '%s\\n' "
        f"{shlex.quote(f'import sys; sys.path.insert(0, {REMOTE_PYDEPS_DIR!r})')} "
        f"> {shlex.quote(GUEST_PYDEPS_PTH)}"
    )


def guest_verify_command() -> list[str]:
    """Import every shipped package in the guest and print what resolved.

    Printing ``__file__`` is the point: an import alone cannot distinguish the
    bundle from a copy the guest image already had, and that ambiguity is the
    bug this module was written to remove.
    """
    bundled_names = [import_name for import_name, _ in PYDEPS_DISTRIBUTIONS]
    body = (
        "import importlib\n"
        "import os\n"
        f"names = {list(PYDEPS_VERIFY_IMPORTS)!r}\n"
        f"bundled = {bundled_names!r}\n"
        f"root = os.path.realpath({REMOTE_PYDEPS_DIR!r})\n"
        "for name in names:\n"
        "    module = importlib.import_module(name)\n"
        "    resolved = os.path.realpath(getattr(module, '__file__', '') or '')\n"
        "    if name in bundled and not (resolved == root or resolved.startswith(root + os.sep)):\n"
        "        raise RuntimeError('%s resolved outside bundle: %s' % (name, resolved))\n"
        "    print(name, getattr(module, '__version__', '?'), "
        "getattr(module, '__file__', '?'))\n"
    )
    return ["/usr/bin/python3", "-c", body]


def guest_manifest_check_command() -> list[str]:
    """Print the guest's installed manifest, or nothing when there is none."""
    return ["bash", "-lc", f"cat {shlex.quote(REMOTE_PYDEPS_MANIFEST)} 2>/dev/null || true"]


def guest_manifest_is_current(raw_output: str) -> bool:
    try:
        installed = json.loads((raw_output or "").strip())
    except (json.JSONDecodeError, TypeError):
        return False
    return installed == expected_manifest()


def _tarball_manifest(tarball: Path) -> dict:
    try:
        with tarfile.open(tarball, "r:gz") as archive:
            manifest_file = archive.extractfile(PYDEPS_MANIFEST)
            manifest = json.load(manifest_file) if manifest_file else None
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read Python dependency bundle {tarball}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Python dependency bundle has no valid manifest: {tarball}")
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guest_exec(base_url: str, command: list[str], timeout: int, what: str,
                request_client=requests) -> dict:
    response = request_client.post(
        f"{base_url.rstrip('/')}/execute",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"command": command, "shell": False}),
        timeout=(10, timeout),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"{what}: status={response.status_code}: {str(response.text)[-2000:]}"
        )
    try:
        result = response.json()
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"{what}: guest returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{what}: guest returned {type(result).__name__}, expected object")
    try:
        returncode = int(result.get("returncode", 0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{what}: guest returned invalid returncode") from exc
    if returncode != 0:
        raise RuntimeError(
            "{}: returncode={} stdout_tail={} stderr_tail={}".format(
                what,
                returncode,
                str(result.get("output", ""))[-2000:],
                str(result.get("error", ""))[-2000:],
            )
        )
    return result


def install_and_verify_python_deps(
    base_url: str,
    sudo_password: str,
    *,
    request_client=requests,
) -> dict:
    """Deliver the bundle to a ready guest and verify bytes, manifest, and imports.

    Providers call this synchronously from their ready hook, before DesktopEnv
    creates the setup controller.  It intentionally has no opt-out: setup must
    never race or depend on a later agent bootstrap.
    """
    tarball = ensure_python_deps_tarball()
    manifest = _tarball_manifest(tarball)
    archive_sha256 = _sha256_file(tarball)
    base_url = str(base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("Guest base URL is required for Python dependency delivery")

    with tarball.open("rb") as handle:
        upload = request_client.post(
            f"{base_url}/setup/upload",
            files={
                "file_data": (
                    tarball.name,
                    handle,
                    "application/octet-stream",
                )
            },
            data={"file_path": REMOTE_PYDEPS_TARBALL},
            timeout=(10, 600),
        )
    if upload.status_code != 200:
        raise RuntimeError(
            "upload Python dependency bundle: "
            f"status={upload.status_code}: {str(upload.text)[-2000:]}"
        )

    hash_result = _guest_exec(
        base_url,
        ["/usr/bin/sha256sum", REMOTE_PYDEPS_TARBALL],
        timeout=300,
        what="hash Python dependency bundle",
        request_client=request_client,
    )
    remote_hash = str(hash_result.get("output", "")).strip().split(maxsplit=1)
    if not remote_hash or remote_hash[0].lower() != archive_sha256:
        actual = remote_hash[0] if remote_hash else "<missing>"
        raise RuntimeError(
            "Python dependency bundle hash mismatch after upload: "
            f"expected {archive_sha256}, got {actual}"
        )

    extract = guest_extract_script(REMOTE_PYDEPS_TARBALL)
    privileged_command = (
        "printf '%s\\n' "
        f"{shlex.quote(sudo_password)} | sudo -S -p '' bash -c "
        f"{shlex.quote(extract)}"
    )
    _guest_exec(
        base_url,
        ["bash", "-lc", privileged_command],
        timeout=600,
        what="unpack Python dependency bundle",
        request_client=request_client,
    )
    verify_result = _guest_exec(
        base_url,
        guest_verify_command(),
        timeout=300,
        what="verify Python dependency bundle imports",
        request_client=request_client,
    )
    manifest_result = _guest_exec(
        base_url,
        ["/bin/cat", REMOTE_PYDEPS_MANIFEST],
        timeout=60,
        what="verify Python dependency bundle manifest",
        request_client=request_client,
    )
    try:
        installed_manifest = json.loads(str(manifest_result.get("output", "")).strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Guest Python dependency manifest is not valid JSON") from exc
    if installed_manifest != manifest:
        raise RuntimeError(
            "Guest Python dependency manifest mismatch: "
            f"expected {manifest!r}, got {installed_manifest!r}"
        )
    return {
        "tarball": str(tarball),
        "sha256": archive_sha256,
        "manifest": manifest,
        "verification_output": str(verify_result.get("output", "")).strip(),
    }
