import unittest
import numpy as np
from profile import BPM1D

class PropagationChecks(unittest.TestCase):
    def test_sampling_and_initial_plane(self):
        for nz in (7,200,203):
            model=BPM1D(nx=128,nz=nz,x_span=400e-6,z_span=1e-4,input_waist=20e-6)
            x,z,intensity=model.propagate()
            self.assertEqual(intensity.shape,(min(nz+1,200),128))
            self.assertAlmostEqual(z[0],0)
            self.assertAlmostEqual(z[-1],model.z_span)
            np.testing.assert_allclose(intensity[0],abs(model._gaussian_input())**2)
            self.assertTrue(np.isfinite(intensity).all())
    def test_gaussian_diffraction(self):
        model=BPM1D(nx=1024,nz=300,x_span=500e-6,z_span=1e-3,input_waist=10e-6)
        x,z,I=model.propagate()
        width=2*np.sqrt(np.sum(x*x*I[-1])/np.sum(I[-1]))
        zr=np.pi*model.n0*model.input_waist**2/model.wavelength
        expected=model.input_waist*np.sqrt(1+(z[-1]/zr)**2)
        self.assertAlmostEqual(width/expected,1,places=4)

if __name__ == '__main__': unittest.main()
