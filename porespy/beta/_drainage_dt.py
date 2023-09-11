import numpy as np
from edt import edt
from porespy.filters import trim_disconnected_blobs
from porespy import settings
from porespy.tools import get_tqdm, Results


__all__ = [
    'drainage_dt',
]


tqdm = get_tqdm()


def drainage_dt(im, inlets, residual=None):
    r"""
    This is a reference implementation of drainage using distance transforms
    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    bins = np.unique(dt[im])[::-1]
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    for i, r in enumerate(tqdm(bins, **settings.tqdm)):
        seeds = dt >= r
        seeds = trim_disconnected_blobs(seeds, inlets=inlets)
        if not np.any(seeds):
            continue
        nwp = edt(~seeds, parallel=settings.ncores) < r
        if residual is not None:
            blobs = trim_disconnected_blobs(residual, inlets=nwp)
            seeds = dt >= r
            seeds = trim_disconnected_blobs(seeds, inlets=blobs + inlets)
            nwp = edt(~seeds, parallel=settings.ncores) < r
        mask = nwp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
    if residual is not None:
        im_seq[im_seq > 0] += 1
        im_seq[residual] = 1
        im_size[residual] = -np.inf
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


if __name__ == '__main__':
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    from edt import edt

    cm = copy(plt.cm.plasma)
    cm.set_under('darkgrey')
    cm.set_over('k')

    # %%
    im = ps.generators.blobs(
        shape=[200, 200, 200], porosity=0.7, blobiness=2, seed=1)
    im = ps.filters.fill_blind_pores(im)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    outlets = np.zeros_like(im)
    outlets[-1, ...] = True
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180

    drn = drainage_dt(im=im, inlets=inlets)
    pc = -2*sigma*np.cos(np.radians(theta))/(drn.im_size*voxel_size)
    pc_curve1 = ps.metrics.pc_map_to_pc_curve(pc, im=im)

    trapped = ps.filters.find_trapped_regions(drn.im_seq, outlets=outlets)
    pc[trapped] = np.inf
    pc_curve2 = ps.metrics.pc_map_to_pc_curve(pc, im=im)

    lt = ps.filters.local_thickness(im)
    nwpr = lt > np.unique(lt)[-7]
    drn = drainage_dt(im=im, inlets=inlets, residual=nwpr)
    pc = -2*sigma*np.cos(np.radians(theta))/(drn.im_size*voxel_size)
    pc_curve3 = ps.metrics.pc_map_to_pc_curve(pc, im=im)

    # %%
    fig, ax = plt.subplots()
    ax.semilogx(pc_curve1.pc, pc_curve1.snwp, 'b-o', label='No Trapping')
    ax.semilogx(pc_curve2.pc, pc_curve2.snwp, 'r-o', label='With Trapping')
    ax.semilogx(pc_curve3.pc, pc_curve3.snwp, 'g-o', label='With Residual')
    ax.semilogx([0, pc_curve3.pc.max()], [nwpr.sum()/im.sum(), nwpr.sum()/im.sum()], 'k-')
    ax.legend(loc='lower right')
    if im.ndim == 2:
        fig, ax = plt.subplots()
        ax.imshow(np.log10(pc))
