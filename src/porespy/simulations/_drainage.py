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


def drainage(
    im,
    pc,
    dt=None,
    inlets=None,
    outlets=None,
    residual=None,
    bins: int = 25,
    return_sizes: bool = False,
):
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
    inlets : ndarray, optional
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. If not specified then access limitations
        are not applied so the result is essentially a local thickness filter.
    outlets : ndarray, optional
        Similar to ``inlets`` except defining the outlets. This image is used
        to assess trapping. If not provided then trapping is ignored,
        otherwise a mask indicating which voxels were trapped is included
        among the returned data.
    residual : ndarray, optional
        A boolean array indicating the locations of any residual invading
        phase. This is added to the intermediate image prior to trimming
        disconnected clusters, so will create connections to some clusters
        that would otherwise be removed. The residual phase is indicated
        in the final image by ``-np.inf`` values, since these are invaded at
        all applied capillary pressures.
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in ``pc``. If a list is given, each value in the list is used
        in ascending order.
    return_sizes : bool
        If `True` then an array containing the size of the sphere which first
        overlapped each pixel is returned. This array is not computed by default
        as computing it increases computation time.

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
        im_size    If `return_sizes` was set to `True`, then a numpy array with
                   each voxel containing the radius of the sphere, in voxels, that
                   first overlapped it.
        im_trapped A numpy array with ``True`` values indicating trapped voxels
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
        return_sizes=return_sizes,
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
    return_sizes=False,
):
    im = np.array(im, dtype=bool)

    if dt is None:
        dt = edt(im)

    # if inlets is None:
    #     inlets = np.zeros_like(im)
    #     inlets[0, ...] = True

    if outlets is not None:
        outlets = outlets*im

    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(bins, int):  # Use values in pc for invasion steps
        vmax = pc[pc < np.inf].max()
        vmin = pc[im][pc[im] > -np.inf].min()
        Ps = np.linspace(vmin, vmax*1.1, bins)
    else:
        Ps = np.unique(bins)  # To ensure they are in ascending order

    # Initialize empty arrays to accumulate results of each loop
    pc_inv = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)

    # Begin IBOP algorithm
    strel = ball(1) if im.ndim == 3 else disk(1)
    for p in tqdm(Ps, **settings.tqdm):
        # Find all locations in image invadable at current pressure
        invadable = (pc <= p)*im
        # Trim locations not connected to the inlets
        if inlets is not None:
            invadable = trim_disconnected_blobs(
                im=invadable,
                inlets=inlets,
                strel=strel,
            )
        # Isolate only newly found locations to speed up inserting
        temp = invadable*(~seeds)
        # Find (i, j, k) coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += invadable
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords]
        # Insert spheres of given radii at new locations
        pc_inv = _insert_disks_at_points_parallel(
            im=pc_inv,
            coords=np.vstack(coords),
            radii=radii.astype(int),
            v=p,
            smooth=True,
            overwrite=False,
        )
        if return_sizes:
            pc_inv = _insert_disks_at_points_parallel(
                im=pc_inv,
                coords=np.vstack(coords),
                radii=radii.astype(int),
                v=max(radii),
                smooth=True,
                overwrite=False,
            )
        # Deal with impact of residual, if present
        if residual is not None:
            # Find residual connected to current invasion front
            inv_temp = (pc_inv > 0)
            if np.any(inv_temp):
                # Find invadable pixels connected to surviving residual
                temp = trim_disconnected_blobs(residual, inv_temp, strel=strel)*~inv_temp
                if np.any(temp):
                    # Trim invadable pixels not connected to residual
                    new_seeds = trim_disconnected_blobs(invadable, temp, strel=strel)
                    # Find (i, j, k) coordinates of new locations
                    coords = np.where(new_seeds)
                    # Add new locations to list of invaded locations
                    seeds += new_seeds
                    # Extract the local size of sphere to insert at each new location
                    radii = dt[coords].astype(int)
                    # Insert spheres of given radii at new locations
                    pc_inv = _insert_disks_at_points_parallel(
                        im=pc_inv,
                        coords=np.vstack(coords),
                        radii=radii.astype(int),
                        v=p,
                        smooth=True,
                        overwrite=False,
                    )
                    if return_sizes:
                        pc_inv = _insert_disks_at_points_parallel(
                            im=pc_inv,
                            coords=np.vstack(coords),
                            radii=radii.astype(int),
                            v=max(radii),
                            smooth=True,
                            overwrite=False,
                        )

    # Set uninvaded voxels to inf
    pc_inv[(pc_inv == 0)*im] = np.inf

    # Add residual if given
    if residual is not None:
        pc_inv[residual] = -np.inf

    # Analyze trapping and adjust computed images accordingly
    trapped = None
    if outlets is not None:
        seq = pc_to_seq(pc_inv, im=im, mode='drainage')
        trapped = find_trapped_regions(seq=seq, outlets=outlets)
        trapped[seq == -1] = True
        pc_inv[trapped] = np.inf
        if residual is not None:  # Re-add residual to inv
            pc_inv[residual] = -np.inf

    # Initialize results object
    results = Results()
    results.im_satn = pc_to_satn(pc=pc_inv, im=im, mode='drainage')
    results.im_pc = pc_inv
    if trapped is not None:
        results.im_trapped = trapped
    results.pc, results.snwp = pc_curve(im=im, pc=pc_inv)
    return results


ibop.__doc__ = drainage.__doc__


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    from pyedt import edt

    # %% Run this cell to regenerate the variables in drainage
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
        g=0,
        voxelsize=1e-4,
    )

    # %% Run different drainage simulations
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
    drn5 = ps.simulations.drainage(
        im=im,
        pc=pc,
    )

    # %% Visualize the invasion configurations for each scenario
    if plots:
        cmap = copy(plt.cm.plasma)
        cmap.set_under(color='black')
        cmap.set_over(color='grey')
        # cmap.set_bad(color='white')
        vmax = pc.max()*2
        fig, ax = plt.subplots(2, 3, facecolor=bg)

        tmp = np.copy(drn1.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[0][0].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[0][0].set_title("No trapping, no residual")

        tmp = np.copy(drn2.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[0][1].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[0][1].set_title("With trapping, no residual")

        tmp = np.copy(drn3.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[1][0].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[1][0].set_title("No trapping, with residual")

        tmp = np.copy(drn4.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[1][1].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[1][1].set_title("With trapping, with residual")

        tmp = np.copy(drn5.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[0][2].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[0][2].set_title("No access limitations")

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
