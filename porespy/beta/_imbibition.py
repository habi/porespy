import numpy as np
from porespy.filters import (
    local_thickness,
    find_trapped_regions,
    size_to_satn,
    size_to_seq,
    seq_to_satn,
    trim_disconnected_blobs,
    find_disconnected_voxels,
)
from porespy.tools import (
    Results,
    get_tqdm,
    make_contiguous,
)
from porespy import settings
from edt import edt
from numba import njit, prange


__all__ = [
    'imbibition',
]


tqdm = get_tqdm()


def imbibition(im, pc, inlets=None, residual=None, bins=25):
    r"""
    Performs an imbibition simulation using image-based sphere insertion

    Parameters
    ----------
    im : ndarray
        The image of the porous materials with void indicated by ``True``
    inlets : ndarray
        An image the same shape as ``im`` with ``True`` values indicating the
        wetting fluid inlet(s).  If ``None`` then the wetting film is able to
        appear anywhere within the domain.
    residual : ndarray, optional
        A boolean mask the same shape as ``im`` with ``True`` values
        indicating to locations of residual wetting phase.
    bins : int
        The number of points to generate

    Notes
    -----
    The simulation proceeds as though the non-wetting phase pressure is very
    high and is slowly lowered. Then imbibition occurs into the smallest
    accessible regions at each step. Blind or inaccessible pores are
    assumed to be filled with wetting phase.

    Examples
    --------

    """
    dt = np.around(edt(im), decimals=0).astype(int)

    pc[~im] = -np.inf
    if isinstance(bins, int):
        bins = np.logspace(1, np.log10(pc.max()), bins)
    bins = np.unique(bins)[::-1]

    im_pc = np.zeros_like(im, dtype=float)
    im_seq = np.zeros_like(im, dtype=int)
    for i, p in tqdm(enumerate(bins), **settings.tqdm):
        seeds = (pc <= p)*im
        coords = np.where(seeds)
        # Reduce number of insert sites by
        # seeds += new_seeds
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords]
        # Insert spheres at new locations of given radii
        im_pc = _insert_disks_npoints_nradii_1value_parallel(
            im=im_pc, coords=coords, radii=radii, v=p, overwrite=True)
        im_seq = _insert_disks_npoints_nradii_1value_parallel(
            im=im_seq, coords=coords, radii=radii, v=i+1, overwrite=True)
        if inlets is not None:
            tmp = trim_disconnected_blobs(im*(im_pc > p), inlets=inlets)
            im_pc[~tmp] = p
            im_seq[~tmp] = i + 1

    # Collect data in a Results object
    result = Results()
    im_seq[~im] = 0
    im_pc[~im] = 0
    im_seq = make_contiguous(im_seq)
    result.im_pc = im_pc
    result.im_seq = im_seq

    return result


@njit(parallel=True)
def _insert_disks_npoints_nradii_1value_parallel(
    im,
    coords,
    radii,
    v,
    overwrite=False,
    smooth=False,
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        for row in prange(len(coords[0])):
            i, j = coords[0][row], coords[1][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if (R <= r)*(~smooth) or (R < r)*(smooth):
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    else:
        xlim, ylim, zlim = im.shape
        for row in prange(len(coords[0])):
            i, j, k = coords[0][row], coords[1][row], coords[2][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(k-r, k+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if (R <= r)*(~smooth) or (R < r)*(smooth):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


if __name__ == '__main__':
    import porespy as ps
    import matplotlib.pyplot as plt
    import numpy as np
    from edt import edt
    from copy import copy

    cm = copy(plt.cm.turbo)
    cm.set_under('grey')
    cm.set_over('k')

    # %%
    im = ~ps.generators.random_spheres([400, 400], r=25, clearance=25, seed=0, edges='extended')
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    inlets[-1, :] = True
    outlets = np.zeros_like(im)
    outlets[:, 0] = True
    outlets[:, -1] = True
    vx = 1e-5
    pc = 2*0.072/(edt(im)*vx)

    # Perform imbibition with no trapping
    imb1 = imbibition(im=im, pc=pc)
    imb2 = imbibition(im=im, pc=pc, inlets=inlets)


    fig, ax = plt.subplots(2, 2)

    tmp = np.copy(imb1.im_seq)
    tmp[~im] = -1
    ax[0][0].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=-0.01)

    tmp = np.copy(imb2.im_seq)
    tmp[~im] = -1
    ax[0][1].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=-0.01)

    seq = ps.filters.pc_to_seq(pc=imb1.im_pc, im=im, mode='imbibition')
    np.all(seq == imb1.im_seq)
    seq = ps.filters.find_trapped_regions(imb1.im_seq, outlets=outlets, return_mask=False)
    satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='imbibition')
    satn[satn < 0] = 2
    satn[~im] = -1
    ax[1][0].imshow(satn, origin='lower', interpolation='none', cmap=cm, vmin=-0.01, vmax=1.0)

    seq = ps.filters.pc_to_seq(pc=imb2.im_pc, im=im, mode='imbibition')
    np.all(seq == imb2.im_seq)
    seq = ps.filters.find_trapped_regions(imb2.im_seq, outlets=outlets, return_mask=False)
    satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='imbibition')
    satn[satn < 0] = 2
    satn[~im] = -1
    ax[1][1].imshow(satn, origin='lower', interpolation='none', cmap=cm, vmin=-0.01, vmax=1.0)

    # %%
    fig, ax = plt.subplots()

    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb1.im_pc, im=im, seq=imb1.im_seq, mode='imbibition')
    ax.plot(pccurve.pc, pccurve.snwp, c='tab:blue', marker='o')

    seq = ps.filters.find_trapped_regions(imb1.im_seq, outlets=outlets, return_mask=False)
    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb1.im_pc, im=im, seq=seq, mode='imbibition')
    ax.plot(pccurve.pc, pccurve.snwp, c='tab:green', marker='o')

    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb2.im_pc, im=im, seq=imb2.im_seq, mode='imbibition')
    ax.plot(pccurve.pc, pccurve.snwp, c='tab:red', marker='o')

    seq = ps.filters.find_trapped_regions(imb2.im_seq, outlets=outlets, return_mask=False)
    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb2.im_pc, im=im, seq=seq, mode='imbibition')
    ax.plot(pccurve.pc, pccurve.snwp, c='tab:orange', marker='o')

    from porespy import beta
    drn = beta.drainage(im=im, pc=pc, inlets=inlets)
    seq = ps.filters.pc_to_seq(drn.im_pc, im=im, mode='drainage')
    pccurve = ps.metrics.pc_map_to_pc_curve(pc=drn.im_pc, im=im, mode='drainage')
    ax.plot(pccurve.pc, pccurve.snwp, c='k', marker='o')





















