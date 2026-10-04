# Optical beam propagation — 1D FFT-BPM

Ali Yaghi · OPTIQ academic photonics project.

The repository name is historical: the present implementation is a **physics-based split-step FFT beam-propagation model**, not a trained neural network. It supports a homogeneous medium, a slab waveguide and a two-guide coupler with a NiceGUI interface.

```bash
python -m pip install -r requirements.txt
python main.py
```

For numerical checks without launching the interface:
```bash
python -m unittest -v test_bpm
```

`profile.py` contains `BPM1D`. All model inputs use SI units; the GUI labels indicate its micrometre/millimetre conversions. The paraxial scalar approximation, periodic FFT grid and edge absorber limit the model. Check convergence in grid size and propagation step for each application. This is not a full-vector Maxwell solver.

The cleaned-up implementation fixes saved-plane indexing for small and non-divisible step counts: the initial field is now at z=0 and the last plane at the requested propagation distance. The physics core no longer imports NiceGUI.

The `docs/` notebook retains the original course context. See [VALIDATION.md](VALIDATION.md), [CONTRIBUTING.md](CONTRIBUTING.md), and the existing MIT licence. Other projects: [portfolio](https://github.com/yaghi-ali/scientific-projects).
