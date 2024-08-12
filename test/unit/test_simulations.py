import pytest
import numpy as np
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import disk, ball, skeletonize_3d
from skimage.util import random_noise
from scipy.stats import norm
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


ps.settings.tqdm['disable'] = True


class SimulationsTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100],
                                      blobiness=2,
                                      seed=0,
                                      porosity=0.499829)
        assert self.im.sum()/self.im.size == 0.499829
        self.im_dt = edt(self.im)

    def test_drainage_with_gravity(self):
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7066, seed=0)
        # im = ps.generators.blobs(shape=[400, 400], porosity=0.7066, seed=2)
        assert im.sum()/im.size == 0.7066
        dt = edt(im)
        pc = ps.simulations.capillary_transform(
            im=im,
            dt=dt,
            sigma=0.072,
            theta=180,
            rho_nwp=997,
            voxelsize=1e0,
            g=0,
        )
        np.testing.assert_approx_equal(pc[im].max(), 0.144)
        # With inaccessible regions, resulting in inf in some voxels (uninvaded)
        drn = ps.simulations.drainage(pc=pc, im=im)
        np.testing.assert_approx_equal(drn.im_pc.max(), np.inf)

        # After filling inaccessible voxels
        im2 = ps.filters.fill_blind_pores(im, conn=2*im.ndim, surface=True)
        assert im2.sum()/im2.size < 0.7066
        dt2 = edt(im2)
        pc2 = ps.simulations.capillary_transform(
            im=im2,
            dt=dt2,
            sigma=0.072,
            theta=180,
            rho_nwp=997,
            voxelsize=1e0,
            g=0,
        )
        drn2 = ps.simulations.drainage(pc=pc2*im2, im=im2)
        np.testing.assert_approx_equal(drn2.im_pc[im2].max(), 0.14630939404790602)

        pc3 = ps.simulations.capillary_transform(
            im=im2,
            dt=dt2,
            sigma=0.072,
            theta=180,
            rho_nwp=997,
            voxelsize=1e-4,
            g=0,
        )
        drn3 = ps.simulations.drainage(pc=pc3, im=im2)
        np.testing.assert_approx_equal(drn3.im_pc.max(), 1463.0940030415854)

    def test_gdd(self):
        from porespy import beta
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.703276, seed=1)
        assert im.sum()/im.size == 0.703276
        res = beta.tortuosity_gdd(im=im, scale_factor=3)

        np.testing.assert_approx_equal(res.tau[0], 1.3940746215566113, significant=5)
        np.testing.assert_approx_equal(res.tau[1], 1.4540191053977147, significant=5)
        np.testing.assert_approx_equal(res.tau[2], 1.4319705063316652, significant=5)

    def test_gdd_dataframe(self):
        from porespy import beta
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.703276, seed=1)
        df = beta.chunks_to_dataframe(im=im, scale_factor=3)
        assert len(df.iloc[:, 0]) == 54
        assert df.columns[0] == 'Throat Number'
        assert df.columns[1] == 'Tortuosity'
        assert df.columns[2] == 'Diffusive Conductance'
        assert df.columns[3] == 'Porosity'

        np.testing.assert_array_almost_equal(np.array(df.iloc[:, 1]),
                                             np.array([1.329061, 1.288042, 1.411449,
                                                       1.273172, 1.46565,  1.294553,
                                                       1.553851, 1.299077, 1.417645,
                                                       1.332902, 1.365739, 1.37725,
                                                       1.408786, 1.279847, 1.365632,
                                                       1.31547,  1.425769, 1.417447,
                                                       1.399028, 1.262936, 1.311554,
                                                       1.447341, 1.504881, 1.196132,
                                                       1.508335, 1.273323, 1.361239,
                                                       1.334868, 1.443466, 1.328017,
                                                       1.564574, 1.264049, 1.504227,
                                                       1.471079, 1.366275, 1.349767,
                                                       1.473522, 1.34229,  1.258255,
                                                       1.266575, 1.488935, 1.260175,
                                                       1.471782, 1.295077, 1.463962,
                                                       1.494004, 1.551485, 1.363379,
                                                       1.474238, 1.311737, 1.483244,
                                                       1.287134, 1.735833, 1.38633],
                                                      ),
                                             decimal=4)


if __name__ == '__main__':
    t = SimulationsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
