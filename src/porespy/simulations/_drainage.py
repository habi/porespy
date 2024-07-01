import numpy as np
from skimage.morphology import ball, disk
from porespy import settings
from porespy.metrics import pc_curve
from porespy.tools import (
    _insert_disks_at_points,
    _insert_disks_at_points_parallel,
    get_tqdm,
    Results,
)
from porespy.filters import (
    trim_disconnected_blobs,
    find_trapped_regions,
    pc_to_satn,
    pc_to_seq,
)
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


__all__ = [
    'drainage',
    'ibop',
]


tqdm = get_tqdm()


def drainage(im, pc, dt=None, inlets=None, outlets=None, residual=None, bins=25):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    pc : ndarray
        An array containing precomputed capillary pressure values in each
        voxel. This can include gravity effects or not.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes. If not specified it is
        assumed that the invading phase enters from the bottom (x=0).
    outlets : ndarray, optional
        Similar to ``inlets`` except defining the outlets. This image is used
        to assess trapping. If not provided then trapping is ignored,
        otherwise a mask indicating which voxels were trapped is included
        amoung the returned data.
    residual : ndarray, optional
        A boolean array indicating the locations of any residual invading
        phase. This is added to the intermediate image prior to trimming
        disconnected clusters, so will create connections to some clusters
        that would otherwise be removed. The residual phase is indicated
        in the final image by ``-np.inf`` values, since there are invaded at
        all applied capillary pressures.
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the ``pc``.  If a list is given, each value in the list is used
        directly in order.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== ============================================================
        Attribute  Description
        ========== ============================================================
        im_pc      A numpy array with each voxel value indicating the
                   capillary pressure at which it was invaded
        im_satn    A numpy array with each voxel value indicating the global
                   saturation value at the point it was invaded
        im_seq     An ndarray with each voxel indicating the step number at
                   which it was first invaded by non-wetting phase
        pc         1D array of capillary pressure values that were applied
        swnp       1D array of non-wetting phase saturations for each applied
                   value of capillary pressure (``pc``).
        ========== ============================================================

    Notes
    -----
    - The direction of gravity is always towards the x=0 axis
    - This algorithm has only been tested for gravity stabilized
      configurations, meaning the more dense fluid is on the bottom.
      Be sure that ``inlets`` are specified accordingly.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/drainage.html>`_
    to view online example.

    """
    results = ibop(
        im=im,
        pc=pc,
        dt=dt,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        bins=bins,
    )
    return results


def ibop(
    im,
    pc,
    dt=None,
    inlets=None,
    outlets=None,
    residual=None,
    bins=25,
):
    r"""
    """
    im = np.array(im, dtype=bool)

    if dt is None:
        dt = edt(im)

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(bins, int):  # Use values in pc for invasion steps
        vmax = pc[pc < np.inf].max()
        vmin = pc[im][pc[im] > -np.inf].min()
        Ps = np.linspace(vmin, vmax*1.1, bins)
    else:
        Ps = bins

    # Initialize empty arrays to accumulate results of each loop
    pc_inv = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)

    # Remove wetting phase blocked from inlets by residual (if present)
    mask = None
    if (residual is not None) and (outlets is not None):
        mask = im * (~residual)
        mask = trim_disconnected_blobs(mask, inlets=inlets)

    # Begin IBOP algorithm
    strel = ball(1) if im.ndim == 3 else disk(1)
    for p in tqdm(Ps, **settings.tqdm):
        p = Ps[i]
        i += 1
        # Find all locations in image invadable at current pressure
        temp = (pc <= p)*im
        # Add residual so that fluid is more easily reconnected
        if residual is not None:
            temp = temp + residual
        # Trim locations not connected to the inlets
        new_seeds = trim_disconnected_blobs(temp, inlets=inlets, strel=strel)
        # Trim locations not connected to the outlet (for trapping)
        if mask is not None:
            new_seeds = new_seeds * mask
        # Isolate only newly found locations to speed up inserting
        temp = new_seeds*(~seeds)
        # Find i,j,k coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += new_seeds
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords].astype(int)
        # Insert spheres of given radii at new locations
        pc_inv = _insert_disks_at_points_parallel(
            im=pc_inv,
            coords=np.vstack(coords),
            radii=radii,
            v=p,
            smooth=True,
        )

    # Set uninvaded voxels to inf
    pc_inv[(pc_inv == 0)*im] = np.inf

    # Add residual if given
    if residual is not None:
        pc_inv[residual] = -np.inf

    trapped = None

    # Analyze trapping and adjust computed images accordingly
    if outlets is not None:
        seq = pc_to_seq(pc_inv, im=im, mode='drainage')
        trapped = find_trapped_regions(seq=seq, outlets=outlets)
        trapped[seq == -1] = True
        pc_inv[trapped] = np.inf
        if residual is not None:  # Re-add residual to inv
            pc_inv[residual] = -1
        satn = pc_to_satn(pc=pc_inv, im=im)


    # Initialize results object
    results = Results()
    results.im_satn = pc_to_satn(pc_inv, im, mode='drainage')
    results.im_pc = pc_inv
    results.im_trapped = trapped

    _pccurve = pc_curve(im=im, pc=pc_inv)
    results.pc = _pccurve.pc
    results.snwp = _pccurve.snwp
    return results


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    from pyedt import edt

    # %% Run this cell to regenerate the variables in drainage
    np.random.seed(6)
    bg = 'white'
    plots = True
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.7,
        blobiness=1.5,
    )
    im = ps.filters.fill_blind_pores(im, surface=True)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True

    lt = ps.filters.local_thickness(im)
    dt = edt(im)
    residual = lt > 25
    bins = 25
    pc = ps.simulations.capillary_transform(
        im=im,
        dt=dt,
        sigma=0.072,
        theta=180,
        rho_nwp=1000,
        rho_wp=0,
        g=9.81,
        voxelsize=1e-4,
    )

    # %% Run 4 different drainage simulations
    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
    )
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
    )
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
    )
    drn4 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
    )

    # %% Visualize the invasion configurations for each scenario
    if plots:
        cmap = copy(plt.cm.plasma)
        cmap.set_under(color='black')
        cmap.set_over(color='grey')
        cmap.set_bad(color='w')
        fig, ax = plt.subplots(2, 2, facecolor=bg)

        tmp = np.copy(drn1.im_pc)
        tmp[~im] = pc.max()*2
        ax[0][0].imshow(tmp, cmap=cmap, vmin=0, vmax=pc.max())
        ax[0][0].set_title("No trapping, no residual")

        tmp = np.copy(drn2.im_pc)
        tmp[~im] = pc.max()*2
        ax[0][1].imshow(tmp, cmap=cmap, vmin=0, vmax=pc.max())
        ax[0][1].set_title("With trapping, no residual")

        tmp = np.copy(drn3.im_pc)
        tmp[~im] = pc.max()*2
        ax[1][0].imshow(tmp, cmap=cmap, vmin=0, vmax=pc.max())
        ax[1][0].set_title("No trapping, with residual")

        tmp = np.copy(drn4.im_pc)
        tmp[~im] = pc.max()*2
        ax[1][1].imshow(tmp, cmap=cmap, vmin=0, vmax=pc.max())
        ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if plots:
        plt.figure(facecolor=bg)
        ax = plt.axes()
        ax.set_facecolor(bg)
        plt.step(drn1.pc, drn1.snwp, 'b-o', where='post',
                 label="No trapping, no residual")
        plt.step(drn2.pc, drn2.snwp, 'r--o', where='post',
                 label="With trapping, no residual")
        plt.step(drn3.pc, drn3.snwp, 'g--o', where='post',
                 label="No trapping, with residual")
        plt.step(drn4.pc, drn4.snwp, 'm--o', where='post',
                 label="With trapping, with residual")
        plt.legend()
