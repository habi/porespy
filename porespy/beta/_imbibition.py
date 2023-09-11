import numpy as np
from porespy.filters import (
    local_thickness,
    find_trapped_regions,
    size_to_satn,
    size_to_seq,
    seq_to_satn,
    pc_to_satn,
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


def imbibition(
    im,
    pc,
    inlets=None,
    residual=None,
    bins=25,
    return_seq=False,
    return_sizes=False,
    return_snwp=False,
):
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
        bins = np.logspace(1, np.log10(pc[im].max()), bins)
    bins = np.unique(bins)[::-1]  # Sort from highest to lowest

    im_pc = np.zeros_like(im, dtype=float)
    if return_seq:
        im_seq = np.zeros_like(im, dtype=int)
    if return_sizes:
        im_size = np.zeros_like(im, dtype=int)
    for i in tqdm(range(len(bins)-1), **settings.tqdm):
        p = bins[i]
        # The following "bracketed" threshold produces exactly the same result
        # as a single threshold `seeds = (pc <= bins[i])*im` without any extra
        # consideration like filling in the missing bits later. Using a bracket
        # is much faster since there are less spheres to insert.
        seeds = (pc <= bins[i])*(pc > bins[i+1])*im
        coords = np.where(seeds)
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords]
        # Insert spheres at new locations of given radii
        im_pc = _insert_disks_npoints_nradii_1value_parallel(
            im=im_pc, coords=coords, radii=radii, v=p, overwrite=True)
        if return_seq:
            im_seq = _insert_disks_npoints_nradii_1value_parallel(
                im=im_seq, coords=coords, radii=radii, v=i+1, overwrite=True)
        if return_sizes:
            try:
                r = radii[0]
            except IndexError:
                r = 0
            im_size = _insert_disks_npoints_nradii_1value_parallel(
                im=im_size, coords=coords, radii=radii, v=r, overwrite=True)
        if inlets is not None:
            tmp = trim_disconnected_blobs(im*(im_pc > p), inlets=inlets)
            im_pc[~tmp] = p
            if return_seq:
                im_seq[~tmp] = i + 1
            if return_sizes:
                im_size[~tmp] = r

    # Collect data in a Results object
    result = Results()
    im_pc[~im] = 0
    result.im_pc = im_pc
    if return_snwp:
        satn = pc_to_satn(pc=im_pc, im=im, mode='imbibition')
        result.im_snwp = satn
    if return_seq:
        im_seq[~im] = 0
        im_seq = make_contiguous(im_seq)
        result.im_seq = im_seq
    if return_sizes:
        im_size[~im] = 0
        result.im_size = im_size
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
    # im = ~ps.generators.random_spheres([200, 200, 200], r=10, clearance=10, seed=0, edges='extended')
    im = ps.generators.blobs([200, 200, 200], porosity=0.65, blobiness=1.5)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    outlets = np.zeros_like(im)
    outlets[-1, ...] = True
    vx = 1e-5
    pc = 2*0.072/(edt(im)*vx)
    pc[~im] = np.inf

    # Perform imbibition with no trapping
    imb1 = imbibition(im=im, pc=pc, return_seq=True, return_sizes=True)
    imb2 = imbibition(im=im, pc=pc, inlets=inlets, return_seq=True, return_sizes=True)

    # %%
    if im.ndim == 2:
        fig, ax = plt.subplots(3, 2)

        tmp = np.copy(imb1.im_size)
        tmp[~im] = -1
        ax[0][0].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=-0.01)
        ax[0][0].set_title("Size")

        tmp = np.copy(imb2.im_size)
        tmp[~im] = -1
        ax[0][1].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=-0.01)
        ax[0][1].set_title("Size")

        tmp = np.copy(imb1.im_seq)
        tmp[~im] = -1
        ax[1][0].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=-0.01)
        ax[1][0].set_title("Sequence")

        tmp = np.copy(imb2.im_seq)
        tmp[~im] = -1
        ax[1][1].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=-0.01)
        ax[1][1].set_title("Sequence")

        seq = ps.filters.pc_to_seq(pc=imb1.im_pc, im=im, mode='imbibition')
        np.sum(seq != imb1.im_seq)
        seq = ps.filters.find_trapped_regions(imb1.im_seq, outlets=outlets, return_mask=False)
        satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='imbibition')
        satn[satn < 0] = 2
        satn[~im] = -1
        ax[2][0].imshow(satn, origin='lower', interpolation='none', cmap=cm, vmin=-0.01, vmax=1.0)
        ax[2][0].set_title("Saturation after Trapping")

        seq = ps.filters.pc_to_seq(pc=imb2.im_pc, im=im, mode='imbibition')
        np.sum(seq != imb2.im_seq)
        seq = ps.filters.find_trapped_regions(imb2.im_seq, outlets=outlets, return_mask=False)
        satn = ps.filters.seq_to_satn(seq=seq, im=im, mode='imbibition')
        satn[satn < 0] = 2
        satn[~im] = -1
        ax[2][1].imshow(satn, origin='lower', interpolation='none', cmap=cm, vmin=-0.01, vmax=1.0)
        ax[2][1].set_title("Saturation after Trapping")

    # %%
    fig, ax = plt.subplots()

    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb1.im_pc, im=im, seq=imb1.im_seq, mode='imbibition')
    ax.semilogx(pccurve.pc, pccurve.snwp, c='tab:blue', marker='o', label='no access limitations, no trapping')

    seq = ps.filters.find_trapped_regions(imb1.im_seq, outlets=outlets, return_mask=False)
    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb1.im_pc, im=im, seq=seq, mode='imbibition')
    ax.semilogx(pccurve.pc, pccurve.snwp, c='tab:green', marker='s', label='no access limitations, with trapping')

    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb2.im_pc, im=im, seq=imb2.im_seq, mode='imbibition')
    ax.semilogx(pccurve.pc, pccurve.snwp, c='tab:red', marker='d', label='with access limitations, no trapping')

    seq = ps.filters.find_trapped_regions(imb2.im_seq, outlets=outlets, return_mask=False)
    pccurve = ps.metrics.pc_map_to_pc_curve(pc=imb2.im_pc, im=im, seq=seq, mode='imbibition')
    ax.semilogx(pccurve.pc, pccurve.snwp, c='tab:orange', marker='^', label='with access limitations, with trapping')

    from porespy import beta
    drn = beta.drainage(im=im, pc=pc, inlets=outlets)
    seq = ps.filters.pc_to_seq(drn.im_pc, im=im, mode='drainage')
    pccurve = ps.metrics.pc_map_to_pc_curve(pc=drn.im_pc, im=im, mode='drainage')
    ax.semilogx(pccurve.pc, pccurve.snwp, c='k', marker='o', label='drainage')
    ax.legend()





















