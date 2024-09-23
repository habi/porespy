import numpy as np
from skimage.morphology import ball, disk
from porespy.filters import (
    find_trapped_regions,
    seq_to_satn,
    trim_disconnected_blobs,
)
from porespy.metrics import pc_map_to_pc_curve
from porespy.tools import (
    Results,
    get_tqdm,
    make_contiguous,
    _insert_disks_at_points,
    _insert_disks_at_points_parallel,
)
from porespy import settings
from edt import edt
from numba import njit, prange


__all__ = [
    'imbibition_dt',
]


tqdm = get_tqdm()


def imbibition_dt(im, inlets=None, residual=None):
    r"""
    This is a reference implementation of imbibition using distance transforms
    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    bins = np.linspace(1, dt.max() + 1, dt.max() + 1, dtype=int)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    for i, r in enumerate(tqdm(bins, **settings.tqdm)):
        seeds = dt >= r
        wp = im*~(edt(~seeds, parallel=settings.ncores) < r)
        if inlets is not None:
            wp = trim_disconnected_blobs(wp, inlets=inlets)
        if residual is not None:
            blobs = trim_disconnected_blobs(residual, inlets=wp)
            seeds = dt >= r
            seeds = trim_disconnected_blobs(seeds, inlets=blobs + inlets)
            wp = im*~(edt(~seeds, parallel=settings.ncores) < r)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1
    if residual is not None:
        im_seq[im_seq > 0] += 1
        im_seq[residual] = 1
        im_size[residual] = np.inf
    results = Results()
    results.im_seq = im_seq
    results.im_size = im_size
    return results
