"""Frequency-angle spectrum of the THz band from a binary kw_THz file (res/kw_THz/Z<pos>.kwb).

Builds the same histograms as f20-dcwin/plot_kw_angle.py does from the text kw_THz files and saves them
in the npz that angular_spectrum() of plot_uppe.ipynb reads (hist, hist_omega, hist_e2, f_edge, th_edge).

The file stores |E(f, kr)|^2 on the Hankel nodes. In the normalized units of the code k = w*t0 and
kappa = kr*cL*t0, the propagation angle is sin(theta) = kappa/k; nodes with kappa >= k (evanescent)
are dropped, as are nearly grazing angles (cos(theta) <= 0.05).
Per (f, theta) bin the mean over the samples that fall into it is stored:
    hist        |E|^2 k^2 sin(theta) cos(theta)   (energy per unit angle)
    hist_omega  |E|^2 k^2 cos(theta)              (energy per solid angle)
    hist_e2     |E|^2
t0 comes from the file header.

Usage: python kw_spectrum.py <in.kwb> <out.npz> [--f-max THz] [--th-max deg] [--df THz] [--dth deg]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

MAGIC = b"UPPEKW01"
HEADER = np.dtype(
    [("magic", "S8"), ("nr", "<i4"), ("nf", "<i4"), ("z_cm", "<f8"), ("t0", "<f8"), ("a0", "<f8"), ("cL", "<f8")]
)
ROWS_PER_BLOCK = 256


def read_kwb(path: Path):
    """Header (dict), f_THz, k_cm, kr_cm and a read-only memmap of E2[nr][nf]."""
    h = np.fromfile(path, dtype=HEADER, count=1)
    if h.size != 1 or h["magic"][0] != MAGIC:
        raise RuntimeError(f"{path}: not a kw_THz file")
    nr, nf = int(h["nr"][0]), int(h["nf"][0])
    head = {k: float(h[k][0]) for k in ("z_cm", "t0", "a0", "cL")}
    head.update(nr=nr, nf=nf)
    off = HEADER.itemsize
    axes = np.fromfile(path, dtype="<f8", count=2 * nf + nr, offset=off)
    if axes.size != 2 * nf + nr:
        raise RuntimeError(f"{path}: truncated axes")
    off += 8 * axes.size
    expected = off + 4 * nr * nf
    size = path.stat().st_size
    if size != expected:
        raise RuntimeError(f"{path}: {size} bytes, expected {expected}")
    e2 = np.memmap(path, dtype="<f4", mode="r", offset=off, shape=(nr, nf))
    return head, axes[:nf], axes[nf : 2 * nf], axes[2 * nf :], e2


def node_dk(kappa: np.ndarray) -> np.ndarray:
    """Hankel spacing of the nodes; plot_kw_angle.py skips the nodes where it is not positive."""
    dk = np.zeros_like(kappa)
    if kappa.size < 2:
        return dk
    dk[1:-1] = 0.5 * (kappa[2:] - kappa[:-2])
    dk[0] = max(kappa[1] - kappa[0], 0.0)
    dk[-1] = max(kappa[-1] - kappa[-2], 0.0)
    dk[(kappa <= 0.0) | (dk <= 0.0)] = 0.0
    return dk


def spectrum(path: Path, f_max: float = 25.0, th_max: float = 30.0, df: float = 0.05, dth: float = 0.1):
    t_start = time.perf_counter()
    head, f, k_cm, kr_cm, e2 = read_kwb(path)
    t0, cL = head["t0"], head["cL"]
    k = k_cm * cL * t0          # = freqs of the code (w*t0)
    kappa = kr_cm * cL * t0     # = kr/a0*cL*t0
    dk = node_dk(kappa)

    # round, not floor: 94.8/0.05 is 1895.9999999999998 in float64.
    nf_bins = int(np.round(f_max / df))
    na_bins = int(np.round(th_max / dth))
    nbin = nf_bins * na_bins
    acc = np.zeros(nbin)
    acc_omega = np.zeros(nbin)
    acc_e2 = np.zeros(nbin)
    cnt = np.zeros(nbin)

    fcols = (f > 0.0) & (f < f_max)
    f_sel = f[fcols]
    k_sel = k[fcols]
    fi_sel = np.floor(f_sel / df).astype(np.int64)
    rows = np.flatnonzero((kappa > 0.0) & (dk > 0.0))
    for b0 in range(0, rows.size, ROWS_PER_BLOCK):
        idx = rows[b0 : b0 + ROWS_PER_BLOCK]
        kap = kappa[idx][:, None]
        e = np.asarray(e2[idx][:, fcols], dtype=np.float64)
        kk = np.broadcast_to(k_sel, e.shape)
        sint = np.clip(kap / kk, 0.0, 1.0)
        cost = np.sqrt(np.maximum(0.0, 1.0 - sint * sint))
        m = (kk > kap) & (cost > 0.05)
        if not m.any():
            continue
        sint = sint[m]
        cost = cost[m]
        kkm = kk[m]
        em = e[m]
        q_omega = em * (kkm * kkm) * cost
        q = q_omega * sint
        th = np.degrees(np.arcsin(sint))
        fi = np.broadcast_to(fi_sel, e.shape)[m]
        ai = np.floor(th / dth).astype(np.int64)
        inside = (fi >= 0) & (fi < nf_bins) & (ai >= 0) & (ai < na_bins)
        flat = fi[inside] * na_bins + ai[inside]
        acc += np.bincount(flat, weights=q[inside], minlength=nbin)
        acc_omega += np.bincount(flat, weights=q_omega[inside], minlength=nbin)
        acc_e2 += np.bincount(flat, weights=em[inside], minlength=nbin)
        cnt += np.bincount(flat, minlength=nbin)

    def mean(a):
        return np.divide(a, cnt, out=np.zeros_like(a), where=cnt > 0.0).reshape(nf_bins, na_bins)

    out = {
        "hist": mean(acc),
        "hist_omega": mean(acc_omega),
        "hist_e2": mean(acc_e2),
        "f_edge": np.arange(nf_bins + 1) * df,
        "th_edge": np.arange(na_bins + 1) * dth,
        "z_cm": np.float64(head["z_cm"]),
        "t0": np.float64(t0),
    }
    print(
        f"{path.name}: z = {head['z_cm']:.4f} cm, {head['nr']} kr nodes x {head['nf']} frequencies, "
        f"{rows.size} propagate, t0 = {t0 * 1e15:.3g} fs, {time.perf_counter() - t_start:.1f} s",
        flush=True,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path, help="res/kw_THz/Z<pos>.kwb")
    p.add_argument("out", type=Path, help="output .npz")
    p.add_argument("--f-max", type=float, default=25.0, help="frequency span, THz (default 25)")
    p.add_argument("--th-max", type=float, default=30.0, help="angle span, degrees (default 30)")
    p.add_argument("--df", type=float, default=0.05, help="frequency bin, THz (default 0.05)")
    p.add_argument("--dth", type=float, default=0.1, help="angle bin, degrees (default 0.1)")
    a = p.parse_args()
    res = spectrum(a.src, a.f_max, a.th_max, a.df, a.dth)
    np.savez_compressed(a.out, **res)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
