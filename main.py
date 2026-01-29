import numpy as np
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nicegui import ui
from profile import BPM1D


# Interface NiceGUI


ui.dark_mode().enable()
ui.label("**1D FFT-BPM Simulation**").classes('text-2xl font-bold text-center mt-3')

with ui.row().classes('w-full justify-center mt-4'):
    profile_sel = ui.select(['Espace libre', 'Guide monomode', 'Coupleur directionnel'],
                            value='Espace libre', label='Structure')
    Lx_in = ui.number('Lx (µm)', value=200.0)
    Lz_in = ui.number('Lz (mm)', value=5.0)
    Nx_in = ui.number('Nx', value=1024)
    Nz_in = ui.number('Nz', value=1000)
    dn_in = ui.number('Δn', value=4e-3)
    wc_in = ui.number('Largeur cœur (µm)', value=10.0)
    sep_in = ui.number('Séparation (µm)', value=25.0)
    w0_in = ui.number('w0 (µm)', value=5.0)
    lambda_in = ui.number('Longueur d’onde λ (µm)', value=1.064, step=0.01)
# --- Zone d'affichage ---

output_container = ui.column().classes('w-full items-center justify-center mt-6')


# Fonction d'exécution


def run_bpm():
    global output_container

    struct = profile_sel.value
    profile = "free" if struct == "Espace libre" else "slab" if struct == "Guide monomode" else "coupler"

    sep_um = sep_in.value
    wc_um = wc_in.value
    if profile == "coupler":
        sep_um = max(sep_um, 2.8 * wc_um)
        delta_n = max(dn_in.value, 3e-3)
        input_center = -0.5 * sep_um
    else:
        delta_n = dn_in.value
        input_center = 0.0

    bpm = BPM1D(
        wavelength=lambda_in.value * 1e-6,
        n0=1.45,
        x_span=Lx_in.value * 1e-6,
        nx=int(Nx_in.value),
        z_span=Lz_in.value * 1e-3,
        nz=int(Nz_in.value),
        profile=profile,
        delta_n=delta_n,
        slab_core_width=wc_um * 1e-6,
        coupler_sep_factor=(sep_um / wc_um if wc_um != 0 else 2.5),
        input_waist=w0_in.value * 1e-6,
        input_center=input_center * 1e-6,
    )

    x, z, I = bpm.propagate()
    x_um, z_mm = x * 1e6, z * 1e3

    fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')
    im = ax.imshow(I.T, extent=[z_mm[0], z_mm[-1], x_um[0], x_um[-1]],
                   aspect='auto', origin='lower', cmap='inferno', interpolation='bilinear')

    ax.set_xlabel('z (mm)', color='white', fontsize=11)
    ax.set_ylabel('x (µm)', color='white', fontsize=11)
    ax.set_title(f'Propagation — {struct}', color='white', fontsize=13)
    for s in ax.spines.values():
        s.set_color('white')

    xticks = np.linspace(z_mm[0], z_mm[-1], 6)
    yticks = np.linspace(x_um[0], x_um[-1], 7)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis='both', which='major', colors='white', length=6, width=1.2, direction='inout', labelsize=9)
    ax.set_xticklabels([f'{v:.1f}' for v in xticks], color='white', fontsize=9)
    ax.set_yticklabels([f'{v:.0f}' for v in yticks], color='white', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Intensity (a.u.)', color='white', fontsize=9)
    cbar.ax.tick_params(colors='white', labelsize=8)
    for label in cbar.ax.get_yticklabels():
        label.set_color('white')

    ax.grid(True, color='gray', alpha=0.25, linewidth=0.4)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    img_uri = f"data:image/png;base64,{img_b64}"

    output_container.clear()
    with output_container:
        ui.image(img_uri).classes('w-[650px] rounded-xl shadow-lg border border-gray-700')

ui.button('Lancer la simulation', on_click=run_bpm).classes('mt-4 bg-primary text-white')
ui.run(title='BPM Simulation')
