"""Step time and results of runs resumed from the same checkpoint, against the first (reference) run.

Each run directory holds log.txt (the program output), res/{Imax,Energy,Nemax,Den}.dat and out.bin, the
checkpoint written when the run stopped. Reported in a Markdown table:
  - mean and median step time without the first step, and the speedup against the reference;
  - mean phase times and the range of R_nl (the "phases" lines; runs of older code have none);
  - ||X - A||_2 / ||A||_2 and max|X - A| / max|A| of E_O and E_T in out.bin;
  - the largest relative difference of the scalar files, entry by entry.

Usage: python bench_compare.py <label>=<run dir> [<label>=<run dir> ...] [--out summary.md]
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import numpy as np

# Header of the checkpoint (src/checkpoint.cpp): 8 int32 after the magic, then iz and clamped_total (int64),
# z, z_flag, Imax (double), nO, nT (uint64), focus, zsic (double), kw_hash (uint64); a uint64 sum at the end.
CKPT_HEADER = np.dtype(
    [("magic", "S8"), ("version", "<i4"), ("nt_all", "<i4"), ("nR_first", "<i4"), ("nR_opt", "<i4"),
     ("nR_THz", "<i4"), ("dc_profile", "<i4"), ("stage", "<i4"), ("kw_next", "<i4"), ("iz", "<i8"),
     ("clamped_total", "<i8"), ("z", "<f8"), ("z_flag", "<f8"), ("Imax", "<f8"), ("nO", "<u8"), ("nT", "<u8"),
     ("focus", "<f8"), ("zsic", "<f8"), ("kw_hash", "<u8")]
)
SCALARS = ("Imax.dat", "Energy.dat", "Nemax.dat", "Den.dat")
PHASES = ("hnkT", "hnkO", "nl", "fft", "lin", "io")
CHUNK = 1 << 22


def read_fields(path: Path):
    """Header and read-only memmaps of E_O and E_T of a checkpoint."""
    h = np.fromfile(path, dtype=CKPT_HEADER, count=1)
    if h.size != 1 or h["magic"][0] != b"UPPECKPT":
        raise RuntimeError(f"{path}: not a checkpoint")
    nO, nT = int(h["nO"][0]), int(h["nT"][0])
    off = CKPT_HEADER.itemsize
    if path.stat().st_size != off + 16 * (nO + nT) + 8:
        raise RuntimeError(f"{path}: size does not match the header")
    E_O = np.memmap(path, dtype="<c16", mode="r", offset=off, shape=(nO,))
    E_T = np.memmap(path, dtype="<c16", mode="r", offset=off + 16 * nO, shape=(nT,))
    return h[0], E_O, E_T


def field_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """||b - a||_2 / ||a||_2 and max|b - a| / max|a|, in chunks."""
    num = den = dmax = amax = 0.0
    for i in range(0, a.size, CHUNK):
        x = np.asarray(a[i : i + CHUNK])
        d = np.abs(np.asarray(b[i : i + CHUNK]) - x)
        ax = np.abs(x)
        num += float(np.dot(d, d))
        den += float(np.dot(ax, ax))
        dmax = max(dmax, float(d.max(initial=0.0)))
        amax = max(amax, float(ax.max(initial=0.0)))
    return (num / den) ** 0.5 if den > 0 else float(num > 0), dmax / amax if amax > 0 else float(dmax > 0)


def table_diff(a: Path, b: Path) -> float:
    x, y = np.loadtxt(a, ndmin=2), np.loadtxt(b, ndmin=2)
    if x.shape != y.shape:
        return float("inf")
    s = np.maximum(np.abs(x), np.abs(y))
    d = np.abs(x - y)
    mask = s > 0
    return float((d[mask] / s[mask]).max(initial=0.0))


def parse_log(path: Path):
    """Step times, mean phase times (dict, empty for older code) and R_nl of every step."""
    steps, phases, rnl = [], {p: [] for p in PHASES}, []
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\s*step time ([0-9.eE+-]+) s", line)
        if m:
            steps.append(float(m.group(1)))
            continue
        if line.lstrip().startswith("phases "):
            tok = line.split()
            for p in PHASES:
                phases[p].append(float(tok[tok.index(p) + 1]))
            rnl.append(int(tok[tok.index("R_nl") + 1]))
    mean = {p: statistics.fmean(v[1:] or v) for p, v in phases.items() if v}
    return steps, mean, rnl


def fmt(x: float) -> str:
    return f"{x:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<label>=<run dir>; the first one is the reference")
    ap.add_argument("--out", type=Path, help="also write the table here")
    args = ap.parse_args()
    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.runs]

    lines = []
    ref_label, ref_dir = runs[0]
    ref_steps, _, _ = parse_log(ref_dir / "log.txt")
    ref_mean = statistics.fmean(ref_steps[1:] or ref_steps)
    ref_h, ref_O, ref_T = read_fields(ref_dir / "out.bin")
    z_last = float(np.loadtxt(ref_dir / "res" / "Imax.dat", ndmin=2)[-1, 0])
    lines.append(f"Resumed at step {int(ref_h['iz']) - len(ref_steps)}, stopped at step {int(ref_h['iz'])} "
                 f"(last step at z = {z_last:.5f} cm); reference: {ref_label}. Step times without the first step.")
    lines.append("")
    lines.append("| run | steps | mean step, s | median, s | speedup | " + " | ".join(PHASES) + " | R_nl |")
    lines.append("|---|---|---|---|---|" + "---|" * len(PHASES) + "---|")
    for label, d in runs:
        steps, ph, rnl = parse_log(d / "log.txt")
        body = steps[1:] or steps
        mean = statistics.fmean(body)
        cols = [label, str(len(steps)), fmt(mean), fmt(statistics.median(body)), f"{ref_mean / mean:.2f}x"]
        cols += [fmt(ph[p]) if p in ph else "" for p in PHASES]
        cols.append(f"{min(rnl)}-{max(rnl)}" if rnl else "")
        lines.append("| " + " | ".join(cols) + " |")

    lines.append("")
    lines.append(f"Differences from {ref_label}:")
    lines.append("")
    lines.append("| run | E_O L2 | E_O max | E_T L2 | E_T max | " + " | ".join(SCALARS) + " |")
    lines.append("|---|---|---|---|---|" + "---|" * len(SCALARS))
    for label, d in runs[1:]:
        h, E_O, E_T = read_fields(d / "out.bin")
        if int(h["iz"]) != int(ref_h["iz"]):
            raise RuntimeError(f"{label} stopped at step {int(h['iz'])}, the reference at {int(ref_h['iz'])}")
        cols = [label, *map(fmt, field_diff(ref_O, E_O)), *map(fmt, field_diff(ref_T, E_T))]
        cols += [fmt(table_diff(ref_dir / "res" / f, d / "res" / f)) for f in SCALARS]
        lines.append("| " + " | ".join(cols) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
