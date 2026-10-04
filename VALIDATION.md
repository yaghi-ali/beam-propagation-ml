# Validation — 2026-10-04

`python -m unittest -v test_bpm`: **2 tests passed**.

The tests check saved-plane indexing for short/non-divisible propagation step counts, finite field values and homogeneous Gaussian-beam propagation against its expected width. The core is tested without starting NiceGUI.

The graphical interface and physical experiments were not exercised. Grid and step convergence remain application-specific. NumPy 2.5.3 and Matplotlib 3.11.2 on Windows/Python 3.12.
