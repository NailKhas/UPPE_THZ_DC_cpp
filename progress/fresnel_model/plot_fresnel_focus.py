#!/usr/bin/env python3
"""Plot axial diffraction for a uniformly illuminated circular pupil and an ideal thin lens."""
import argparse
from datetime import datetime
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/uppe-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def intensity_ratio(z_cm, f_cm, a_cm=0.1, wavelength_cm=744e-7):
    """I(z)/I(f); NumPy sinc uses sin(pi*x)/(pi*x)."""
    z = np.asarray(z_cm, dtype=float)
    if np.any(z <= 0):
        raise ValueError("z must be positive")
    number = a_cm**2 / (wavelength_cm * f_cm)
    return (f_cm / z)**2 * np.sinc(0.5 * number * (f_cm / z - 1))**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    a_cm, wavelength_cm = 0.1, 744e-7
    results, tables = [], []
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                         "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), sharey=True,
                             constrained_layout=True)
    overlay, overlay_axis = plt.subplots(figsize=(9, 5.3), constrained_layout=True)
    for ax, f_cm, color in zip(axes, [20.0, 100.0], ["#215c94", "#c65e15"]):
        number = a_cm**2 / (wavelength_cm * f_cm)
        # The central Fresnel lobe starts at its first pre-focus zero.
        first_zero = f_cm / (1 + 2 / number)
        optimum = minimize_scalar(lambda z: -float(intensity_ratio(z, f_cm)),
                                  bounds=(first_zero, f_cm), method="bounded",
                                  options={"xatol": 1e-11})
        if not optimum.success:
            raise RuntimeError(optimum.message)
        z_peak = float(optimum.x)
        maximum = float(-optimum.fun)
        z = np.unique(np.r_[np.linspace(0.1 * f_cm, 1.7 * f_cm, 4001), f_cm, z_peak])
        ratio = intensity_ratio(z, f_cm)
        assert float(intensity_ratio(f_cm, f_cm)) == 1.0
        assert 0 < z_peak < f_cm and maximum > 1.0
        assert np.isfinite(ratio).all() and np.all(ratio >= 0)
        assert abs(float(np.max(ratio)) - maximum) < 1e-8
        # Independent local check: the geometric focus lies on a falling slope.
        step = f_cm * 1e-5
        derivative = float((intensity_ratio(f_cm + step, f_cm)
                            - intensity_ratio(f_cm - step, f_cm)) / (2 * step))
        assert abs(derivative + 2 / f_cm) < 1e-6
        ax.plot(z, ratio, color=color, lw=1.8)
        ax.axvline(f_cm, color="#777777", ls="--", label=f"Геометрический фокус: {f_cm:g} см")
        ax.axvline(z_peak, color=color, ls=":", alpha=0.8)
        ax.plot(z_peak, maximum, "o", color=color,
                label=f"Максимум: z={z_peak:.3f} см, I/I(f)={maximum:.4f}")
        ax.plot(f_cm, 1, "o", color="#777777", ms=4)
        ax.set(xlabel="z, см", title=f"f = {f_cm:g} см; N = {number:.4f}",
               xlim=(0.1 * f_cm, 1.7 * f_cm), ylim=(0, 1.7))
        ax.legend(loc="upper right", fontsize=8)
        overlay_axis.plot(z / f_cm, ratio, color=color, lw=1.7,
                          label=f"f={f_cm:g} см, N={number:.4f}")
        overlay_axis.plot(z_peak / f_cm, maximum, "o", color=color)
        results.append({"f_cm": f_cm, "fresnel_number": number, "z_peak_cm": z_peak,
                        "z_peak_over_f": z_peak / f_cm, "I_peak_over_If": maximum,
                        "I_focus_over_Iin": (np.pi * number)**2,
                        "I_peak_over_Iin": maximum * (np.pi * number)**2,
                        "d_Iratio_dz_at_f_per_cm": derivative})
        tables.append(np.column_stack([np.full(len(z), f_cm), np.full(len(z), number),
                                       z, ratio, ratio * (np.pi * number)**2]))
    axes[0].set_ylabel(r"$I(z)/I(f)$")
    fig.suptitle("Линейная дифракция: равномерно освещённая круглая апертура\n"
                 "a=1 мм, λ=744 нм; каждая кривая нормирована на собственное I(f)")
    overlay_axis.axvline(1, color="#777777", ls="--", label="Геометрический фокус z/f=1")
    overlay_axis.set(xlabel="z/f", ylabel=r"$I(z)/I(f)$", xlim=(0.1, 1.7), ylim=(0, 1.7))
    overlay_axis.legend(fontsize=9)
    overlay_axis.set_title("Сравнение в относительной координате\na=1 мм, λ=744 нм")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(args.output / f"fresnel_focus_f20_f100.{extension}")
        overlay.savefig(args.output / f"fresnel_focus_overlay.{extension}")
    plt.close(fig)
    plt.close(overlay)
    np.savetxt(args.output / "fresnel_focus.csv", np.concatenate(tables), delimiter=",",
               header="f_cm,N,z_cm,I_over_If,I_over_Iin", comments="", fmt="%.14g")
    metadata = {"model": "linear scalar Fresnel diffraction; uniform circular pupil; paraxial ideal thin lens",
                "a_cm": a_cm, "wavelength_nm": wavelength_cm * 1e7,
                "formula": "(f/z)^2 * sinc^2[(pi*N/2)*(f/z-1)], sinc(x)=sin(x)/x",
                "normalization": "Each curve is divided by its own I(f); amplitudes are not equal incident-intensity comparisons.",
                "created_local": datetime.now().astimezone().isoformat(), "cases": results}
    (args.output / "fresnel_focus.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
