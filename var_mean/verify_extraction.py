"""Prove extracted computation matches the snapshot, including ext.program_id int64."""

import ast
from pathlib import Path

from common import ROOT, digest, save


def main():
    source = ast.parse((ROOT / "upstream/var_mean.py.snapshot").read_text())
    original = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "var_mean_welford_kernel"
    )
    helper_source = Path(
        "/root/FlagGems/src/flag_gems/utils/triton_lang_extension.py"
    ).read_text()
    helper = next(
        n
        for n in ast.parse(helper_source).body
        if isinstance(n, ast.FunctionDef) and n.name == "program_id"
    )
    assert ast.unparse(helper.body[0].value) == "tl.program_id(axis).to(tl.int64)"
    extracted = next(
        n
        for n in ast.parse((ROOT / "tutorial.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
    )
    expected = ast.unparse(ast.Module(body=original.body, type_ignores=[])).replace(
        "ext.program_id(0)", "tl.program_id(0).to(tl.int64)"
    )
    assert expected == ast.unparse(ast.Module(body=extracted.body, type_ignores=[]))
    save(
        ROOT / "results/runs/r2/extraction.json",
        dict(
            passed=True,
            source_sha256=digest(ROOT / "upstream/var_mean.py.snapshot"),
            extracted_sha256=digest(ROOT / "tutorial.py"),
            program_id_helper=ast.unparse(helper),
            helper_source_sha256=__import__("hashlib")
            .sha256(helper_source.encode())
            .hexdigest(),
        ),
    )
    print("PASS: exact kernel AST body after upstream int64 helper substitution")


if __name__ == "__main__":
    main()
