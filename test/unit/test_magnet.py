import pytest
import numpy as np
from scipy import stats as spst
import scipy.ndimage as spim
import porespy as ps
import openpnm as op
import skimage as ski

np.random.seed(1)

ws = op.Workspace()
ws.settings['loglevel'] = 50
ps.settings.tqdm['disable'] = True

class MagnetTest:
    def setup_class(self):
        # Define 2D image
        im2 = ps.generators.blobs([100, 100], porosity=0.6, blobiness=2)
        im2 = ps.filters.fill_blind_pores(im2, conn=8, surface=True)
        # Define 3D image
        im3 = ps.generators.blobs([100, 100, 100], porosity=0.25, blobiness=1)
        im3 = ps.filters.fill_blind_pores(im3, conn=26, surface=True)
        im3 = ps.filters.trim_floating_solid(im3, conn=6, surface=False)
        # assign to self
        self.blobs2D = im2
        self.blobs3D = im3

    def test_return_all(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im)
        # assert im.sum()/im.size == 0.52215
        assert hasattr(magnet, 'network')
        assert hasattr(magnet, 'sk')
        assert hasattr(magnet, 'juncs')
        assert hasattr(magnet, 'throat_area')
        assert isinstance(magnet.network, dict)
        assert isinstance(magnet.sk, np.ndarray)
        assert isinstance(magnet.juncs, np.ndarray)
        assert magnet.throat_area is None

    def test_ensure_correct_sizes_are_returned_2d(self):
        im = self.blobs2D
        magnet = ps.networks.magnet(im)
        mode = spst.mode(magnet.network['pore.inscribed_diameter'],
                         keepdims=False)
        assert mode[0] == 4.0
        D = np.unique(magnet.network['pore.inscribed_diameter'].astype(int))
        assert np.all(D == np.array([2, 4, 5, 6, 7, 8, 10, 11, 12]))

    def test_ensure_correct_sizes_are_returned_3d(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im)
        mode = spst.mode(magnet.network['pore.inscribed_diameter'],
                         keepdims=False)
        assert mode[0] == 6.0
        D = np.unique(magnet.network['pore.inscribed_diameter'].astype(int))
        assert np.all(D == np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test_parallel_skeleton_2d(self):
        im = self.blobs2D
        magnet = ps.networks.magnet(im, parallel=True, divs=4)
        sk = magnet.sk
        assert np.sum(sk) == 1259

    def test_parallel_skeleton_3d(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im, parallel=True, divs=4)
        sk = magnet.sk
        assert np.sum(sk) == 7642

    def test_check_skeleton_health(self):
        im = ps.generators.blobs([100, 100, 100], porosity=0.5, blobiness=1)
        sk = ski.morphology.skeletonize_3d(im).astype('bool')
        n = ps.networks._magnet._check_skeleton_health(sk.astype('bool'))
        assert n == 5

    def test_junctions(self):
        im = self.blobs3D
        mode = "maximum filter"
        magnet = ps.networks.magnet(im, throat_junctions=mode)
        assert np.sum(magnet.juncs) == 1583
        mode = "fast marching"
        magnet = ps.networks.magnet(im, throat_junctions=mode)
        assert np.sum(magnet.juncs) == 1491

    def test_throat_area(self):
        im = self.blobs3D
        magnet = ps.networks.magnet(im, throat_area=True)
        D = np.unique(magnet.network['throat.equivalent_diameter'].astype(int))
        d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 20, 21, 28, 29]
        assert np.all(D == np.array(d))
        assert np.isclose(np.sum(magnet.throat_area), 35412.917135930744)


if __name__ == '__main__':
    t = MagnetTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
