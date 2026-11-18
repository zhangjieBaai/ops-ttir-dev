"""Reject the system FlagTree environment; optionally verify installed wheel records."""

import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import sys
from functools import lru_cache
from pathlib import Path

import triton
from triton._C import libtriton

EXPECTED_PREFIX = Path("/root/venvs/ir-ops-dev")


@lru_cache(maxsize=1)
def check_environment():
    assert Path(sys.prefix) == EXPECTED_PREFIX, "Use /root/venvs/ir-ops-dev/bin/python"
    assert triton.__version__ == "3.7.0", "Experiment requires Triton 3.7.0"
    dist = metadata.distribution("triton")
    assert dist.version == "3.7.0"
    assert Path(triton.__file__).is_relative_to(EXPECTED_PREFIX)
    assert Path(libtriton.__file__).is_relative_to(EXPECTED_PREFIX)
    assert all(
        "/root/FlagTree" not in p and "/usr/local/lib/python3.12/dist-packages" not in p
        for p in sys.path
    )
    forbidden = [
        d.metadata["Name"]
        for d in metadata.distributions()
        if any(s in d.metadata["Name"].lower() for s in ["flagtree", "flagprism"])
    ]
    assert not forbidden, forbidden
    assert all(
        importlib.util.find_spec(name) is None for name in ["flagtree", "flagprism"]
    )
    package = Path(triton.__file__).parent
    markers = [
        str(p.relative_to(package))
        for p in package.rglob("*.py")
        if any(word in p.read_text().lower() for word in ["flagtree", "flagprism"])
    ]
    assert not markers, markers
    assert not any(
        name.startswith(("flagtree", "flagprism", "triton._flagprism"))
        for name in sys.modules
    )
    return dict(
        python=sys.executable,
        prefix=sys.prefix,
        triton=triton.__version__,
        triton_path=triton.__file__,
        libtriton_path=libtriton.__file__,
        include_system_site_packages=False,
        sys_path=list(sys.path),
        flagtree_flagprism_distributions=forbidden,
        flagtree_flagprism_source_markers=markers,
        installer=dist.read_text("INSTALLER"),
        direct_url=dist.read_text("direct_url.json"),
    )


def verify_distribution_files():
    """Check content against the installed pip wheel's RECORD (not a remote origin attestation)."""
    result = dict(check_environment())
    dist = metadata.distribution("triton")
    checked = 0
    for name, digest, _ in csv.reader(io.StringIO(dist.read_text("RECORD"))):
        if not digest:
            continue
        algorithm, expected = digest.split("=", 1)
        path = Path(dist.locate_file(name))
        actual = (
            base64.urlsafe_b64encode(hashlib.new(algorithm, path.read_bytes()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert actual == expected, f"Installed Triton file differs from RECORD: {name}"
        checked += 1
    result["verified_record_entries"] = checked
    return result
