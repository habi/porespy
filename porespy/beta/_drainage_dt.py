import numpy as np
from edt import edt
from porespy.filters import trim_disconnected_blobs
from porespy import settings
from porespy.tools import get_tqdm, Results
tqdm = get_tqdm()


def drainage_dt(im, pc, inlets):
    r"""
    This is a reference implementation of IBSI drainage using distance transforms
    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    bins = np.unique(dt[im])[::-1]
    im_pc = np.zeros_like(im, dtype=float)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=int)
    for i, r in enumerate(tqdm(bins, **settings.tqdm)):
        seeds = dt >= r
        seeds = trim_disconnected_blobs(seeds, inlets=inlets)
        if np.any(seeds):
            nwp = edt(~seeds, parallel=settings.ncores) < r
            mask = nwp*(im_pc == 0)
            p = np.amin(pc[dt == r])
            im_pc[mask] = p
            im_size[mask] = r
            im_seq[mask] = i + 1
    results = Results()
    results.im_pc = im_pc
    results.im_seq = im_seq
    results.im_size = im_size
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
        shape=[300, 300, 300], porosity=0.7, blobiness=1.5, seed=0)
    im = ps.filters.fill_blind_pores(im)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    dt = edt(im)
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    pc = -2*sigma*np.cos(np.radians(theta))/(dt*voxel_size)

    drn = drainage_dt(im=im, pc=pc, inlets=inlets)
    pc_curve = ps.metrics.pc_map_to_pc_curve(drn.im_pc, im=im)

    fig, ax = plt.subplots()
    ax.semilogx(pc_curve.pc, pc_curve.snwp, 'b-o')
    if im.ndim == 2:
        fig, ax = plt.subplots()
        ax.imshow(np.log10(drn.im_pc))






















