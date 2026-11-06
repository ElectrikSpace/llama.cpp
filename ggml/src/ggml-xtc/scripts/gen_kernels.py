#!/usr/bin/env python3
#
# Generate ahead-of-time XTC kernels and a manifest for the ggml-xtc backend.
#
# Example:
#   python3 gen_kernels.py --output-dir /tmp/xtc_kernels --m 4 --n 32 --k 512
#

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import xtc.graphs.xtc.op as O
from xtc.backends.mlir import Backend


def build_mul_mat_kernel(output_dir: Path, m: int, n: int, k: int) -> None:
    """Build F32 matmul kernel for ggml MUL_MAT with ne1=m, ne0=n, ne10=k."""
    output_dir.mkdir(parents=True, exist_ok=True)

    a = O.tensor((m, k), "float32", name="A")
    b = O.tensor((k, n), "float32", name="B")
    with O.graph(name="matmul") as gb:
        O.matmul(a, b, name="matmul")
    graph = gb.graph
    impl = Backend(graph)

    sch = impl.get_scheduler()
    sched = sch.schedule()

    dump_base = output_dir / "matmul"
    comp = impl.get_compiler(
        shared_lib=True,
        dump_file=str(dump_base),
        print_source_ir=False,
        print_transformed_ir=False,
        print_lowered_ir=False,
        print_assembly=False,
    )
    code = StringIO()
    with redirect_stderr(code):
        module = comp.compile(sched)

    lib_path = Path(module.file_name)
    if not lib_path.is_absolute():
        lib_path = (output_dir / lib_path.name).resolve()

    manifest = output_dir / "manifest.csv"
    line = (
        f"MUL_MAT,F32,F32,{n},{m},{k},1,1,1,1,"
        f"{lib_path.name},{module.payload_name}\n"
    )
    with manifest.open("w", encoding="utf-8") as outf:
        outf.write("# op,type0,type1,ne0,ne1,ne10,ne02,ne03,ne12,ne13,library,symbol\n")
        outf.write(line)

    print(f"Wrote {lib_path}")
    print(f"Wrote {manifest}")
    print(code.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate XTC kernels for ggml-xtc")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=4, help="ggml ne1 / XTC I")
    parser.add_argument("--n", type=int, default=32, help="ggml ne0 / XTC J")
    parser.add_argument("--k", type=int, default=512, help="ggml ne10 / XTC K")
    args = parser.parse_args()

    build_mul_mat_kernel(args.output_dir, args.m, args.n, args.k)


if __name__ == "__main__":
    main()
