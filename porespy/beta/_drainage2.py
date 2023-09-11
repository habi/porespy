import numpy as np
from edt import edt
from numba import njit, prange
from porespy.filters import (
    trim_disconnected_blobs,
    pc_to_satn,
    pc_to_seq,
)
from porespy import settings
from porespy.tools import get_tqdm, Results
tqdm = get_tqdm()


__all__ = [
    'drainage',
    'elevation_map',
    'drainage_dt',
]


def elevation_map(im_or_shape, voxel_size=1, axis=0):
    r"""
    Generate a image of distances from given axis

    Parameters
    ----------
    im_or_shape : ndarray or list
        This dictates the shape of the output image. If an image is supplied, then
        it's shape is used. Otherwise, the shape should be supplied as a N-D long
        list of the shape for each axis (i.e. `[200, 200]` or `[300, 300, 300]`).
    voxel_size : scalar, optional, default is 1
        The size of the voxels in physical units (i.e. `100e-6` would be 100 um per
        voxel side). If not given that 1 is used, so the returned image is in units
        of voxels.
    axis : int, optional, default is 0
        The direction along which the height is calculated.  The default is 0, which
        is the 'x-axis'.

    Returns
    -------
    elevation : ndarray
        A numpy array of the specified shape with the values in each voxel indicating
        the height of that voxel from the beginning of the specified axis.

    See Also
    --------
    ramp

    """
    if len(im_or_shape) <= 3:
        im = np.zeros(*im_or_shape, dtype=bool)
    else:
        im = im_or_shape
    im = np.swapaxes(im, 0, axis)
    a = np.arange(0, im.shape[0])
    b = np.reshape(a, [im.shape[0], 1, 1])
    c = np.tile(b, (1, *im.shape[1:]))
    c = c*voxel_size
    h = np.swapaxes(c, 0, axis)
    return h


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


def drainage(im, pc, inlets=None, residual=None, bins=25, return_seq=False, return_snwp=True):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    pc : ndarray
        An array containing the capillary pressure space.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes. If not specified it is
        assumed that the invading phase enters from the bottom (x=0).
    residual : ndarray
        A boolean array the same shape a `im` with `True` values indicating the
        location of trapped non-wetting phase (i.e. invading phase).  Note that
        any values of `-inf` in `pc` will also be treated as residual non-wetting
        phase.
    bins : int or array_like (default =  `None`)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the ``pc``.  If a list is given, each value in the list is used
        directly *in order*.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_pc      A numpy array with each voxel value indicating the
                   capillary pressure at which it was invaded
        im_snwp    if `return_snwp` is try, a numpy array with each voxel value
                   indicating the global non-wetting phase saturation value at the
                   point it was invaded.
        ========== =================================================================

    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    pc[~im] = np.inf

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if isinstance(bins, int):
        vals = np.unique(pc)
        vals = vals[~np.isinf(vals)]
        bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), bins)

    if len(bins) == 1:
        bins = [bins[0], bins[0]]

    # Digitize pc
    pc_vals = np.copy(pc)
    pc = np.digitize(pc, bins=bins)
    pc[~im] = 0
    bins = np.unique(pc[im])

    inv_pc = np.zeros_like(im, dtype=float)

    count = 0
    for i in tqdm(range(1, len(bins)), **settings.tqdm):
        p = bins[i]
        seeds = (pc <= bins[i])*im  # Find all locations invadable at current pc
        # Trim locations not connected to the inlets
        seeds = trim_disconnected_blobs(seeds, inlets=inlets)
        # shell = seeds*(pc >= bins[i-1])  # Isolate shell to reduce number of insertions
        coords = np.where(seeds)  # Find i,j,k coordinates of new locations
        radii = dt[coords]  # Extract the local size of sphere to insert
        inv_pc = _insert_disks_npoints_nradii_1value_parallel(  # Insert spheres
            im=inv_pc, coords=coords, radii=radii, v=p)
        # core = seeds*(~shell)*(inv_pc == 0)  # Fill any core not overlapped by spheres
        # inv_pc[core] = p
        if residual is not None:  # Add residual and check for newly connected seeds
            seeds = (pc <= bins[i])*im  # Find all seeds again
            # Find residual clusters connected to invading fluid
            temp = trim_disconnected_blobs((inv_pc > 0)+(residual > 0), inlets=inlets)
            temp = temp*(residual > 0)  # Keep residual connected to invasion front
            if np.any(temp):  # temp can be empty, so this should be skipped
                # Find seeds connected to clusters of surviving residual
                seeds = trim_disconnected_blobs(seeds, inlets=temp)
                coords = np.where(seeds)  # Find i,j,k coordinates of new locations
                radii = dt[coords]  # Extract the local size of sphere to insert
                inv_pc = _insert_disks_npoints_nradii_1value_parallel(  # Insert spheres
                    im=inv_pc, coords=coords, radii=radii, v=p)
        count += 1

    inv_pc[(inv_pc == 0)*im] = np.inf  # Set uninvaded voxels to inf

    if residual is not None:
        inv_pc[residual] = -np.inf

    results = Results()  # Initialize results object and attach arrays
    if return_snwp:
        satn = pc_to_satn(pc=inv_pc, im=im)
        results.im_snwp = satn
    if return_seq:
        seq = pc_to_seq(inv_pc, im=im, mode='drainage')
        results.im_seq = seq
    results.im_pc = inv_pc
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


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    from edt import edt
    cm = copy(plt.cm.plasma)
    cm.set_under('darkgrey')
    cm.set_over('k')

    # %% Add gravity
    im = ps.generators.blobs(
        shape=[200, 200, 200], porosity=0.7, blobiness=1.5, seed=0)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    dt = edt(im)
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = 1000
    g = 9.81

    pc = -2*sigma*np.cos(np.radians(theta))/(dt*voxel_size)
    drn1 = drainage(im=im, pc=pc, inlets=inlets, bins=50)
    pc_curve1 = ps.metrics.pc_map_to_pc_curve(drn1.im_pc, im=im)

    h = elevation_map(im, voxel_size=voxel_size)
    pc = pc + delta_rho*g*h
    drn2 = drainage(im=im, pc=pc, inlets=inlets, bins=50)
    pc_curve2 = ps.metrics.pc_map_to_pc_curve(drn2.im_pc, im=im)

    fig, ax = plt.subplots()
    ax.step(np.log10(pc_curve1.pc), pc_curve1.snwp, where='post', label='No Gravity')
    ax.step(np.log10(pc_curve2.pc), pc_curve2.snwp, where='post', label='With Gravity')
    ax.set_xlabel('log(Capillary Pressure [Pa])')
    ax.set_ylabel('Non-wetting Phase Saturation')
    ax.legend(loc='lower right')

    # %% Add residual nwp
    # im = ~ps.generators.random_spheres(
    #     [400, 400], r=25, clearance=25, seed=1, edges='extended')
    im = ps.generators.blobs(
        shape=[400, 400], porosity=0.7, blobiness=1.5, seed=0)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    outlets = np.zeros_like(im)
    outlets[-1, ...] = True
    dt = edt(im)
    voxel_size = 1e-4
    sigma = 0.072
    pc = 2*sigma/(dt*voxel_size)

    # Generate some residual blobs of nwp
    drn1 = drainage(im=im, pc=pc, bins=np.logspace(1, 4, 50), return_seq=True)
    trapped = ps.filters.find_trapped_regions(seq=drn1.im_seq, outlets=outlets)
    drn2 = drainage(im=im, pc=pc, residual=trapped, bins=np.logspace(1, 4, 50))
    if im.ndim == 2:
        fig, ax = plt.subplots(1, 2)
        tmp = np.copy(drn1.im_snwp)
        tmp[tmp == -1] = 2
        tmp[~im] = -1
        ax[0].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=0, vmax=1)
        ax[0].axis(False)

        tmp = np.copy(drn2.im_snwp)
        tmp[tmp == -1] = 2
        tmp[~im] = -1
        ax[1].imshow(tmp, origin='lower', interpolation='none', cmap=cm, vmin=0, vmax=1)
        ax[1].axis(False)

    # seq = ps.filters.satn_to_seq(drn1.im_snwp)
    # pc_curve1 = ps.metrics.pc_curve(im=im, pc=drn1.im_pc, seq=seq)
    pc_curve1 = ps.metrics.pc_map_to_pc_curve(drn1.im_pc, im=im)

    # seq = ps.filters.satn_to_seq(drn2.im_snwp)
    # pc_curve2 = ps.metrics.pc_curve(im=im, pc=drn2.im_pc, seq=seq)
    pc_curve2 = ps.metrics.pc_map_to_pc_curve(drn2.im_pc, im=im)

    fig, ax = plt.subplots(figsize=[5.5, 5])
    ax.semilogx(pc_curve1.pc, pc_curve1.snwp, marker='o', color='tab:red', label='No residual')
    ax.semilogx(pc_curve2.pc, pc_curve2.snwp, marker='o', color='tab:blue', label='With residual')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([10, 1000])
    ax.set_xlabel('$P_C [Pa]$')
    ax.set_ylabel('Non-Wetting Phase Saturation')
    ax.legend(loc='lower right');
