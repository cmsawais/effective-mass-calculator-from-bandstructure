"""
effective_mass.py
-----------------
Compute electron and hole effective masses from a VASP-style BAND.dat file
(typically produced by VASPKIT, task 211/213) using a parabolic fit around
the band extrema:

        E(k) ≈ E0 + (ħ² / 2m*) (k - k0)²

so that

        m* / m0  =  ħ² / (2 m0 a)  =  3.80998 eV·Å² / a

where  a = d²E/(2 dk²)  is the curvature obtained from a polynomial fit
(units: eV when energy is in eV and k is in 1/Å).

Usage
-----
    python effective_mass.py BAND.dat                     # auto-detect VBM/CBM
    python effective_mass.py BAND.dat --efermi 0.0
    python effective_mass.py BAND.dat --npoints 7         # fitting window
    python effective_mass.py BAND.dat --plot              # show fits
    python effective_mass.py BAND.dat --order 2           # quadratic (default)
                                                          # use 4 for non-parabolic
    python effective_mass.py BAND.dat --kunit 2pi/a --alat 5.65   # convert units

Author: drafted by Claude, based on the EMT slides by G. Ozgur (SMU, 2003)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# ħ² / (2 m_e) in eV · Å²  ->  see e.g. Ashcroft & Mermin, App. B
HBAR2_OVER_2ME_EVA2 = 3.80998212  # eV * Å²


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class BandData:
    """Holds parsed band-structure data."""
    k: np.ndarray            # shape (Nk,)          k-path coordinate
    energies: np.ndarray     # shape (Nk, Nbands)   band energies (eV)
    nbands: int
    nkpts: int


@dataclass
class FitResult:
    """Result of a single parabolic fit."""
    band_index: int
    extremum: str            # "CBM" or "VBM"
    k0: float                # k at extremum (1/Å)
    E0: float                # energy at extremum (eV)
    a: float                 # curvature coefficient (eV·Å²); m* = ħ²/(2 m0 a) * m0
    m_eff_over_m0: float     # signed effective mass / m_e
    rms_eV: float            # RMS error of the fit (eV)
    k_fit: np.ndarray        # k points actually used in the fit
    E_fit: np.ndarray        # E points actually used in the fit
    E_model: np.ndarray      # fitted parabola evaluated on k_fit


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_band_dat(path: Path) -> BandData:
    """
    Parse a BAND.dat file.

    Two common formats are supported:

      (A) VASPKIT 'BAND.dat' (default for vaspkit task 211):
          # NKPTS & NBANDS:  Nk  Nb
          # Band-Index   1
          k1  E1
          k2  E2
          ...
          <blank line>
          # Band-Index   2
          k1  E1
          ...

      (B) Multi-column format (some other tools / vaspkit options):
          # k   E1   E2   E3 ...
          k1  E1_1  E1_2  ...
          k2  E2_1  E2_2  ...
    """
    text = Path(path).read_text()
    raw_lines = text.splitlines()

    # Remove leading whitespace; keep blank lines (they are block separators in format A)
    lines = [ln.rstrip() for ln in raw_lines]

    # Decide between formats by counting columns of a typical data line
    sample_cols = 0
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        sample_cols = len(s.split())
        break

    if sample_cols == 0:
        raise ValueError(f"No numeric data found in {path}")

    if sample_cols == 2:
        return _parse_vaspkit_blocks(lines)
    else:
        return _parse_multicolumn(lines)


def _parse_vaspkit_blocks(lines: list[str]) -> BandData:
    """Format A: blocks separated by blank lines, each preceded by '# Band-Index ...'."""
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for ln in lines:
        s = ln.strip()
        if not s:
            if current:
                blocks.append(current)
                current = []
            continue
        if s.startswith("#"):
            # New band header forces a flush as well
            if current:
                blocks.append(current)
                current = []
            continue
        parts = s.split()
        try:
            k_val = float(parts[0])
            e_val = float(parts[1])
        except (ValueError, IndexError):
            continue
        current.append((k_val, e_val))
    if current:
        blocks.append(current)

    if not blocks:
        raise ValueError("Could not parse any band blocks (vaspkit format).")

    # Sanity check: all blocks must have the same length
    nk = len(blocks[0])
    for i, b in enumerate(blocks):
        if len(b) != nk:
            raise ValueError(
                f"Inconsistent number of k-points: band 1 has {nk}, "
                f"band {i+1} has {len(b)}."
            )

    k = np.array([p[0] for p in blocks[0]], dtype=float)
    energies = np.array(
        [[blocks[b][i][1] for b in range(len(blocks))] for i in range(nk)],
        dtype=float,
    )
    return BandData(k=k, energies=energies, nbands=energies.shape[1], nkpts=nk)


def _parse_multicolumn(lines: list[str]) -> BandData:
    """Format B: every non-comment line is 'k  E1  E2 ...'."""
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(row)

    if not rows:
        raise ValueError("Could not parse any data rows (multi-column format).")

    # Allow trailing blank columns; trim to common width
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=float)

    k = arr[:, 0]
    energies = arr[:, 1:]
    return BandData(k=k, energies=energies, nbands=energies.shape[1], nkpts=len(k))


# ---------------------------------------------------------------------------
# Band extrema and effective-mass fits
# ---------------------------------------------------------------------------
def locate_band_edges(
    bd: BandData,
    efermi: float | None = None,
    tol: float = 1e-3,
) -> tuple[int, int]:
    """
    Identify the indices of the valence band (highest occupied) and conduction
    band (lowest unoccupied).

    The reference is `efermi` (eV). If None, we assume bands have already been
    shifted so that E_F = 0 (this is what VASPKIT does by default).

    Returns
    -------
    (vb_index, cb_index) : 0-based band indices
    """
    if efermi is None:
        efermi = 0.0

    # max energy reached by each band along the path
    band_max = bd.energies.max(axis=0)
    band_min = bd.energies.min(axis=0)

    # Valence band: the band whose maximum is closest to (and not above) E_F
    # Conduction band: the band whose minimum is closest to (and not below) E_F
    vb_candidates = np.where(band_max <= efermi + tol)[0]
    cb_candidates = np.where(band_min >= efermi - tol)[0]

    if len(vb_candidates) == 0 or len(cb_candidates) == 0:
        # Fallback: just sort by min energy and split at E_F
        # Pick highest band whose *max* is below E_F + small slack
        order = np.argsort(band_max)
        below = [i for i in order if band_max[i] < efermi]
        above = [i for i in order if band_min[i] > efermi]
        if not below or not above:
            raise RuntimeError(
                "Could not unambiguously identify VBM/CBM. "
                "Pass --vbm-band and --cbm-band explicitly."
            )
        vb_index = below[-1]
        cb_index = above[0]
    else:
        vb_index = int(vb_candidates[np.argmax(band_max[vb_candidates])])
        cb_index = int(cb_candidates[np.argmin(band_min[cb_candidates])])

    return vb_index, cb_index


def _select_fit_window(
    k: np.ndarray,
    E: np.ndarray,
    i0: int,
    npoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick `npoints` points centered on index i0 (clipped to array bounds).
    The fit window must be at least 3 points wide (need 2nd derivative).
    """
    if npoints < 3:
        raise ValueError("npoints must be >= 3 for a parabolic fit.")
    half = npoints // 2
    lo = max(0, i0 - half)
    hi = min(len(k), i0 + half + 1)
    # If we hit a boundary, expand on the other side to keep the window symmetric in count
    if hi - lo < npoints:
        if lo == 0:
            hi = min(len(k), lo + npoints)
        elif hi == len(k):
            lo = max(0, hi - npoints)
    return k[lo:hi], E[lo:hi]


def fit_effective_mass(
    k: np.ndarray,
    E: np.ndarray,
    extremum: str,
    band_index: int,
    npoints: int = 7,
    order: int = 2,
) -> FitResult:
    """
    Fit a polynomial (default quadratic) around the extremum of E(k) and
    extract the curvature, then convert to effective mass.

    Parameters
    ----------
    k, E       : arrays in 1/Å and eV respectively
    extremum   : "CBM" -> minimum, "VBM" -> maximum
    band_index : just for bookkeeping in the result
    npoints    : number of k-points to include in the fit (centered on extremum)
    order      : polynomial order (>= 2). Curvature is taken as 2*c2 of the
                 polynomial expanded around k0 ; for order > 2 this still
                 captures the parabolic component correctly.

    Returns
    -------
    FitResult
    """
    if extremum == "CBM":
        i0 = int(np.argmin(E))
    elif extremum == "VBM":
        i0 = int(np.argmax(E))
    else:
        raise ValueError("extremum must be 'CBM' or 'VBM'.")

    k_fit, E_fit = _select_fit_window(k, E, i0, npoints)
    k0 = k[i0]
    E0 = E[i0]

    # Fit polynomial in (k - k0) so the constant and linear terms have a clear meaning
    dk = k_fit - k0
    # numpy.polyfit returns highest-order coeff first
    coeffs = np.polyfit(dk, E_fit, order)
    # The quadratic coefficient corresponds to c2 in  E = ... + c2 * dk² + c1 * dk + c0
    c2 = coeffs[-3]  # works for any order >= 2

    # E ≈ E0 + c2 * (k-k0)²  =>  d²E/dk² = 2*c2  =>  m* = ħ² / (2*c2 ... wait)
    # Standard form: E = E0 + (ħ²/2m*) (k-k0)² so  c2 = ħ²/(2 m*)
    # Therefore m*/m0 = (ħ²/(2 m0)) / c2  = 3.80998 / c2
    if abs(c2) < 1e-12:
        m_eff_over_m0 = float("inf")
    else:
        m_eff_over_m0 = HBAR2_OVER_2ME_EVA2 / c2

    # Evaluate the model on the fit window for diagnostics
    E_model = np.polyval(coeffs, dk)
    rms = float(np.sqrt(np.mean((E_model - E_fit) ** 2)))

    return FitResult(
        band_index=band_index,
        extremum=extremum,
        k0=float(k0),
        E0=float(E0),
        a=float(c2),
        m_eff_over_m0=float(m_eff_over_m0),
        rms_eV=rms,
        k_fit=k_fit,
        E_fit=E_fit,
        E_model=E_model,
    )


# ---------------------------------------------------------------------------
# k-axis unit handling
# ---------------------------------------------------------------------------
def convert_k_to_inv_angstrom(
    k: np.ndarray,
    kunit: str,
    alat: float | None,
) -> np.ndarray:
    """
    Convert k-axis to 1/Å.

    kunit options
    -------------
    "1/A"     : already in 1/Å (typical VASPKIT default for BAND.dat)
    "1/nm"    : multiply by 0.1
    "2pi/a"   : k_path is given in units of 2π/a ; needs --alat (Å)
    "frac"    : pure fractional k-path coordinate ; needs --alat (Å)
                (treated like "2pi/a" with the same conversion)
    """
    if kunit == "1/A":
        return k.astype(float)
    if kunit == "1/nm":
        return 0.1 * k
    if kunit in ("2pi/a", "frac"):
        if alat is None or alat <= 0:
            raise ValueError(
                f"--alat must be provided (in Å) when --kunit={kunit}"
            )
        return (2.0 * np.pi / alat) * k
    raise ValueError(f"Unknown --kunit value: {kunit}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute electron/hole effective masses from a "
                    "VASP/VASPKIT BAND.dat file via parabolic fitting."
    )
    p.add_argument("band_file", type=Path, help="Path to BAND.dat")
    p.add_argument(
        "--efermi", type=float, default=None,
        help="Fermi level in eV. Default: assume the file is already "
             "shifted so E_F = 0 (VASPKIT default).",
    )
    p.add_argument(
        "--vbm-band", type=int, default=None,
        help="0-based band index of the VBM (skip auto-detection).",
    )
    p.add_argument(
        "--cbm-band", type=int, default=None,
        help="0-based band index of the CBM (skip auto-detection).",
    )
    p.add_argument(
        "--npoints", type=int, default=7,
        help="Number of k-points centered on the extremum used for the fit "
             "(default: 7). Use a small window so the parabolic approximation "
             "holds — see Eq. (12) in the EMT slides.",
    )
    p.add_argument(
        "--order", type=int, default=2,
        help="Polynomial order (default: 2 = pure parabola). "
             "Use 4 for slight non-parabolicity; the curvature is still "
             "extracted as the quadratic coefficient.",
    )
    p.add_argument(
        "--kunit", choices=["1/A", "1/nm", "2pi/a", "frac"], default="1/A",
        help="Units of the k-column in BAND.dat (default: 1/Å).",
    )
    p.add_argument(
        "--alat", type=float, default=None,
        help="Lattice parameter in Å (needed only for --kunit=2pi/a or frac).",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Show a matplotlib figure with the bands and parabolic fits.",
    )
    p.add_argument(
        "--save-plot", type=Path, default=None,
        help="Save the diagnostic plot to this path (PNG/PDF).",
    )

    args = p.parse_args(argv)

    if not args.band_file.is_file():
        print(f"ERROR: file not found: {args.band_file}", file=sys.stderr)
        return 2

    # 1. Parse
    bd = parse_band_dat(args.band_file)
    print(f"Parsed {args.band_file}:  {bd.nkpts} k-points, {bd.nbands} bands")

    # 2. Convert k-axis to 1/Å
    k_invA = convert_k_to_inv_angstrom(bd.k, args.kunit, args.alat)

    # 3. Identify the VBM / CBM bands
    if args.vbm_band is not None and args.cbm_band is not None:
        vb_idx, cb_idx = args.vbm_band, args.cbm_band
        print(f"Using user-specified bands: VBM=#{vb_idx}, CBM=#{cb_idx}")
    else:
        vb_idx, cb_idx = locate_band_edges(bd, efermi=args.efermi)
        print(f"Auto-detected band edges:   VBM=#{vb_idx}, CBM=#{cb_idx}")

    # 4. Parabolic fits
    cb_fit = fit_effective_mass(
        k_invA, bd.energies[:, cb_idx],
        extremum="CBM", band_index=cb_idx,
        npoints=args.npoints, order=args.order,
    )
    vb_fit = fit_effective_mass(
        k_invA, bd.energies[:, vb_idx],
        extremum="VBM", band_index=vb_idx,
        npoints=args.npoints, order=args.order,
    )

    # 5. Report
    egap = cb_fit.E0 - vb_fit.E0
    direct = abs(cb_fit.k0 - vb_fit.k0) < 1e-4
    print()
    print("=" * 64)
    print("  EFFECTIVE MASS RESULTS  (parabolic fit, "
          f"window = {args.npoints} points, order = {args.order})")
    print("=" * 64)
    _print_fit("Electron (CBM)", cb_fit)
    _print_fit("Hole     (VBM)", vb_fit)
    print("-" * 64)
    print(f"  Band gap E_g  = {egap:.4f} eV   "
          f"({'direct' if direct else 'indirect'})")
    print("=" * 64)

    # 6. Optional plot
    if args.plot or args.save_plot:
        _plot(bd, k_invA, vb_idx, cb_idx, vb_fit, cb_fit,
              show=args.plot, savepath=args.save_plot)

    return 0


def _print_fit(label: str, fit: FitResult) -> None:
    # Convention: report electron mass as +m*, hole mass as |m*| (positive),
    # because hole = absence of electron with negative m*. Eq. (12b) in slides.
    if fit.extremum == "VBM":
        reported = abs(fit.m_eff_over_m0)
        sign_note = "  (|m*|, hole convention)"
    else:
        reported = fit.m_eff_over_m0
        sign_note = ""
    print(f"  {label}:  band #{fit.band_index}")
    print(f"      k0 = {fit.k0:+.5f} 1/Å,   E0 = {fit.E0:+.5f} eV")
    print(f"      curvature a = {fit.a:+.5f} eV·Å²")
    print(f"      m*/m0 = {reported:+.5f}{sign_note}")
    print(f"      RMS of fit = {fit.rms_eV*1e3:.3f} meV")


def _plot(bd, k_invA, vb_idx, cb_idx, vb_fit, cb_fit, show=False, savepath=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot all bands faintly
    for b in range(bd.nbands):
        ax.plot(k_invA, bd.energies[:, b], color="lightgrey", lw=0.8)
    # Highlight VBM / CBM bands
    ax.plot(k_invA, bd.energies[:, cb_idx], color="C0", lw=1.5,
            label=f"CB (band #{cb_idx})")
    ax.plot(k_invA, bd.energies[:, vb_idx], color="C3", lw=1.5,
            label=f"VB (band #{vb_idx})")
    # Overlay parabolic fits
    ax.plot(cb_fit.k_fit, cb_fit.E_model, "o--", color="C0", ms=4,
            label=f"CBM fit, m*/m0 = {cb_fit.m_eff_over_m0:+.3f}")
    ax.plot(vb_fit.k_fit, vb_fit.E_model, "o--", color="C3", ms=4,
            label=f"VBM fit, |m*|/m0 = {abs(vb_fit.m_eff_over_m0):.3f}")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"$k$ (1/Å)")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.set_title("Band structure with parabolic effective-mass fits")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=180)
        print(f"Saved plot to {savepath}")
    if show:
        plt.show()


if __name__ == "__main__":
    raise SystemExit(main())
