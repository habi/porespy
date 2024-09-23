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
    _insert_disks_at_points_parallel,
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
    dt=None,
    inlets=None,
    outlets=None,
    residual=None,
    bins=25,
):
    r"""
    Performs an imbibition simulation using image-based sphere insertion

    Parameters
    ----------
    im : ndarray
        The image of the porous materials with void indicated by ``True``
    pc : ndarray
        An array containing precomputed capillary pressure values in each
        voxel. This can include gravity effects or not. This can be generated
        by ``capillary_transform``.
    inlets : ndarray
        An image the same shape as ``im`` with ``True`` values indicating the
        wetting fluid inlet(s).  If ``None`` then the wetting film is able to
        appear anywhere within the domain.
    residual : ndarray, optional
        A boolean mask the same shape as ``im`` with ``True`` values
        indicating to locations of residual wetting phase.
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in ``pc``. If a list is given, each value in the list is used
        directly in order.

    Returns
    -------
    results : Result Object
        A dataclass-like object with the following attributes:

        ----------- ----------------------------------------------------------------
        Attribute   Description
        ----------- ----------------------------------------------------------------
        im_pc       An ndarray with each voxel indicating the step number at
                    which it was first invaded by wetting phase.
        im_seq      A numpy array with each voxel value indicating the sequence
                    at which it was invaded by the wetting phase.  Values of -1
                    indicate that it was not invaded, either because it was trapped,
                    inaccessbile, or sufficient pressure was not reached.
        im_snwp     A numpy array with each voxel value indicating the global
                    non-wetting phase saturation at the point it was invaded.
        im_trapped  A numpy array with ``True`` values indicating trapped voxels if
                    `outlets` was provided, otherwise will be `None`.
        pc          1D array of capillary pressure values that were applied
        snwp        1D array of non-wetting phase saturations for each applied
                    value of capillary pressure (``pc``).
        ----------- ----------------------------------------------------------------

    Notes
    -----
    The simulation proceeds as though the non-wetting phase pressure is very
    high and is slowly lowered. Then imbibition occurs into the smallest
    accessible regions at each step. Blind or inaccessible pores are
    assumed to be filled with wetting phase.

    Examples
    --------

    """
    if dt is None:
        dt = edt(im)

    pc = np.copy(pc)
    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(bins, int):
        vmax = pc[pc < np.inf].max()
        vmin = pc[im][pc[im] > -np.inf].min()
        Ps = np.logspace(np.log10(np.ceil(vmax)), np.log10(np.floor(vmin)), bins)
    else:
        Ps = np.unique(bins)[::-1]  # To ensure they are in descending order

    # Initialize empty arrays to accumulate results of each loop
    im_pc = np.zeros_like(im, dtype=float)
    im_seq = np.zeros_like(im, dtype=int)
    strel = ball(1) if im.ndim == 3 else disk(1)
    for i in tqdm(range(len(Ps)), **settings.tqdm):
        # This can be made faster if I find a way to get only seeds on edge, so
        # less spheres need to be drawn
        invadable = (pc <= Ps[i])*im
        nwp_mask = np.zeros_like(im, dtype=bool)
        if np.any(invadable):
            coords = np.where(invadable)
            radii = dt[coords].astype(int)
            nwp_mask = _insert_disks_at_points_parallel(
                im=nwp_mask,
                coords=np.vstack(coords),
                radii=radii,
                v=True,
                smooth=True,
                overwrite=True,
            )
        if inlets is not None:
            nwp_mask = ~trim_disconnected_blobs(
                im=(~nwp_mask)*im,
                inlets=inlets,
                strel=strel,
            )
        if residual is not None:
            nwp_mask = nwp_mask * ~residual

        mask = (nwp_mask == 0) * (im_seq == 0) * im
        im_seq[mask] = i
        im_pc[mask] = Ps[i]

    trapped = None
    if outlets is not None:
        if inlets is not None:
            outlets[inlets] = False  # Ensure outlets do not overlap inlets
        trapped = find_trapped_regions(
            im=im, seq=im_seq, outlets=outlets, return_mask=True, method='cluster')
        im_pc[trapped] = -np.inf
        im_seq[trapped] = -1
    satn = seq_to_satn(im=im, seq=im_seq, mode='imbibition')
    pc_curve = pc_map_to_pc_curve(pc=im_pc, im=im, seq=im_seq, mode='imbibition')
    # Collect data in a Results object
    results = Results()
    results.im_seq = im_seq
    results.im_pc = im_pc
    results.im_snwp = satn
    results.im_trapped = trapped
    results.pc = pc_curve.pc
    results.snwp = pc_curve.snwp
    return results


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


# %%

if __name__ == '__main__':
    import porespy as ps
    import matplotlib.pyplot as plt
    import numpy as np
    from edt import edt
    from copy import copy
    ps.visualization.set_mpl_style()

    cm = copy(plt.cm.turbo)
    cm.set_under('grey')
    cm.set_over('k')

    i = np.random.randint(1, 100000)  # bad: 38364, good: 65270, 71698
    print(i)
    im = ps.generators.blobs([500, 500], porosity=0.65, blobiness=2, seed=i)
    im = ps.filters.fill_blind_pores(im, surface=True)

    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    outlets = ps.generators.borders(im.shape, mode='faces')
    lt = ps.filters.local_thickness(im)
    residual = (lt < 8)*im
    pc = ps.filters.capillary_transform(im=im, voxel_size=1e-4)

    imb1 = imbibition(im=im, pc=pc, inlets=inlets)
    imb2 = imbibition(im=im, pc=pc, inlets=inlets, residual=residual, outlets=outlets)

    # %%

    fig, ax = plt.subplots(1, 3)
    imb1.im_pc[~im] = -1
    ax[0].imshow(imb1.im_seq/im, origin='lower', cmap=cm, vmin=0)

    vmax = imb2.im_seq.max()
    ax[1].imshow(imb2.im_seq/im, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    ax[2].semilogx(imb1.pc, imb1.snwp, 'b->', label='imbibition')
    ax[2].semilogx(imb2.pc, imb2.snwp, 'r-<', label='imbibition with trapping')
    ax[2].legend()
