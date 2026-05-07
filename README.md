# Effective Mass Calculator

This repository calculates electron and hole effective masses from a VASP/VASPKIT-style `BAND.dat` file using a parabolic fit around the conduction band minimum (CBM) and valence band maximum (VBM).

The calculation uses:

```text
E(k) = E0 + (hbar^2 / 2m*) (k - k0)^2
```

and reports effective masses in units of the free electron mass, `m0`.

## Files

```text
effective_mass.py      Python script for command-line use
effective_mass.ipynb   Jupyter notebook version for easy step-by-step runs
BAND.dat               Example/input band-structure data file
```

## Requirements

Use Python 3.10 or newer if possible.

Install the required packages:

```bash
pip install numpy matplotlib jupyter
```

`matplotlib` is only needed for plotting. `jupyter` is only needed for the notebook.

## Quick Start: Command Line

Run the script with the included `BAND.dat` file:

```bash
python3 effective_mass.py BAND.dat
```

Show a diagnostic plot:

```bash
python3 effective_mass.py BAND.dat --plot
```

Save the plot to a file:

```bash
python3 effective_mass.py BAND.dat --save-plot effective_mass_fit.png
```

Use a different fitting window:

```bash
python3 effective_mass.py BAND.dat --npoints 7
```

Use a higher-order polynomial fit:

```bash
python3 effective_mass.py BAND.dat --order 4
```

Specify the Fermi level:

```bash
python3 effective_mass.py BAND.dat --efermi 0.0
```

Manually specify the VBM and CBM band indices:

```bash
python3 effective_mass.py BAND.dat --vbm-band 111 --cbm-band 112
```

Band indices are 0-based in this script.

## Quick Start: Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
effective_mass.ipynb
```

Run the notebook cells from top to bottom. In most cases, you only need to edit the **Settings** cell:

```python
band_file = Path("BAND.dat")
efermi = None
vbm_band = None
cbm_band = None
npoints = 7
order = 2
kunit = "1/A"
alat = None
make_plot = True
```

Leave `vbm_band` and `cbm_band` as `None` for automatic band-edge detection. Set them manually if the automatic detection is not appropriate for your material.

At the end of the notebook there is an optional **Band Gap Diagram** section. Run it after the effective-mass cells to plot a clearer band-gap figure with highlighted VBM/CBM bands, the band-gap value, band indices, and editable high-symmetry k-point labels such as Gamma, Delta, X, and M.

## Input Data Format

The script supports two common band-data formats.

### VASPKIT Block Format

This is typical for VASPKIT `BAND.dat` output:

```text
# Band-Index   1
k1  E1
k2  E2

# Band-Index   2
k1  E1
k2  E2
```

### Multi-Column Format

This format has one k-path column followed by band energies:

```text
k   E_band1   E_band2   E_band3
k1  E1        E2        E3
k2  E1        E2        E3
```

Energies should be in eV. The default k-axis unit is `1/Angstrom`.

## k-Axis Units

Use `--kunit` if your k-axis is not already in `1/Angstrom`.

Supported values:

```text
1/A      already in inverse Angstrom
1/nm     inverse nanometer
2pi/a    units of 2*pi/a, requires --alat
frac     fractional k-path coordinate, requires --alat
```

Example:

```bash
python3 effective_mass.py BAND.dat --kunit 2pi/a --alat 5.65
```

## Example Output

For the included `BAND.dat`, the default run gives:

```text
Parsed BAND.dat:  140 k-points, 192 bands
Auto-detected band edges:   VBM=#111, CBM=#112

Electron (CBM):  m*/m0 = +0.48110
Hole     (VBM):  m*/m0 = +2.83014
Band gap E_g  = 5.0599 eV   (indirect)
```

Your values may change if you use a different `BAND.dat`, fitting window, polynomial order, Fermi level, or manually selected bands.

## Notes

- Use a small fitting window near the band extremum so the parabolic approximation remains valid.
- Electron effective mass is reported directly from the CBM curvature.
- Hole effective mass is reported as a positive value using `|m*|`.
- If automatic VBM/CBM detection fails, pass `--vbm-band` and `--cbm-band` manually.


