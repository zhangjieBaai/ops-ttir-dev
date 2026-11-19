"""Compile candidates, retain failures and hash actual ELF machine-code sections."""

import argparse
import json
import struct
import traceback
from pathlib import Path

from common import ROOT, compile_ir, digest, kernel_info, receipt, save_json


def text_hash(cubin):
    # ELF64 little-endian section headers: hash encoded .text bytes, not mnemonics.
    assert cubin[:6] == b"\x7fELF\x02\x01"
    section_offset = struct.unpack_from("<Q", cubin, 40)[0]
    entry_size, count, strings_index = struct.unpack_from("<HHH", cubin, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", cubin, section_offset + i * entry_size)
        for i in range(count)
    ]
    strings = headers[strings_index]
    names = cubin[strings[4] : strings[4] + strings[5]]
    result = {}
    for header in headers:
        name = names[header[0] :].split(b"\0", 1)[0].decode()
        if name.startswith(".text."):
            result[name] = digest(cubin[header[4] : header[4] + header[5]])
    assert result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = [
        "baseline",
        *json.loads((ROOT / "results/candidates_r1_v2.json").read_text())["candidates"],
    ]
    records = {}
    for name in names:
        try:
            ck = compile_ir(name)
            records[name] = dict(
                status="COMPILED",
                **kernel_info(ck, name),
                text_sections=text_hash(ck.asm["cubin"])
            )
        except Exception:
            records[name] = dict(
                status="COMPILE_OR_LOAD_FAILED", traceback=traceback.format_exc()
            )
        print(name, records[name]["status"], flush=True)
    save_json(args.output, dict(**receipt(), kernels=records))


if __name__ == "__main__":
    main()
