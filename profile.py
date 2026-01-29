import numpy as np
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nicegui import ui


# Classe BPM1D
# ceration de la classe avec des parametre deja defini dedans
class BPM1D:
    def __init__(self, wavelength=1.064e-6, n0=1.45, x_span=200e-6, nx=1024,
                 z_span=5e-3, nz=1000, profile="free", delta_n=4e-3,
                 slab_core_width=10e-6, coupler_sep_factor=2.5,
                 input_waist=5e-6, input_center=0.0):
        self.wavelength = wavelength
        self.n0 = n0
        self.x_span = x_span
        self.nx = nx
        self.z_span = z_span
        self.nz = nz
        self.profile = profile
        self.delta_n = delta_n
        self.slab_core_width = slab_core_width
        self.coupler_sep_factor = coupler_sep_factor
        self.input_waist = input_waist
        self.input_center = input_center

        self.dx = self.x_span / self.nx
        self.dz = self.z_span / self.nz
        self.x = (np.arange(self.nx) - self.nx//2) * self.dx
        self.k = 2*np.pi*self.n0 / self.wavelength
        self.kx = 2*np.pi * np.fft.fftfreq(self.nx, d=self.dx)

        self.nx_profile = self._build_index_profile()
        self.V = 2*np.pi/self.wavelength * (self.nx_profile - self.n0)
        self.absorber = self._build_absorber()
        self.diff_op = np.exp(-1j*(self.kx**2)*self.dz/(2*self.k))

    def _build_absorber(self, margin_ratio=0.15, power=4):
        L = self.x_span
        margin = margin_ratio * L
        d = (L/2) - np.abs(self.x)
        w = np.ones_like(self.x)
        edge = d < margin
        xi = np.clip((margin - d[edge]) / margin, 0, 1)
        w[edge] = np.exp(-(xi**power))
        return w

    def _build_index_profile(self):
        """Profil d'indice : free / slab / coupler directionnel."""
        n = np.full_like(self.x, self.n0)

        if self.profile == "slab":
            core = np.abs(self.x) <= (self.slab_core_width / 2)
            n[core] = self.n0 + self.delta_n

        elif self.profile == "coupler":
            # --- Profil gaussien double bien séparé ---
            w = self.slab_core_width
            sep = self.coupler_sep_factor * w
            delta = self.delta_n

            sigma = w / 2.5
            n_left = np.exp(-((self.x + sep / 2) ** 2) / (2 * sigma**2))
            n_right = np.exp(-((self.x - sep / 2) ** 2) / (2 * sigma**2))
            n = self.n0 + delta * (n_left + n_right)

        return n


    def _gaussian_input(self):
        """Injection gaussienne dans le bon cœur."""
        if self.profile == "coupler":
            x0 = -0.5 * self.coupler_sep_factor * self.slab_core_width  # cœur gauche
        else:
            x0 = self.input_center
        return np.exp(-((self.x - x0) ** 2) / (self.input_waist ** 2))


    def propagate(self):
        A = self._gaussian_input().astype(np.complex128)
        P_half = np.exp(1j*self.V*(self.dz/2))
        n_save = 200
        I = np.empty((n_save, self.nx))
        z_samples = np.linspace(0, self.z_span, n_save)
        step = self.nz // n_save
        for i in range(self.nz):
            A *= P_half
            A = np.fft.ifft(self.diff_op * np.fft.fft(A))
            A *= P_half
            A *= self.absorber
            if i % step == 0:
                I[i//step] = np.abs(A)**2
        return self.x, z_samples, I
