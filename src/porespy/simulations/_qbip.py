import logging
import heapq as hq
import numpy as np
import scipy.stats as spst
import numpy.typing as npt
from edt import edt
from numba import njit
from skimage.morphology import skeletonize_3d
from scipy.ndimage import maximum_filter, label
from porespy.generators import ramp
from porespy.filters import (
    region_size,
    seq_to_satn,
    local_thickness,
    flood_func,
)
from porespy.tools import (
    get_tqdm,
    make_contiguous,
    Results,
    ps_round,
    ps_rect,
)


logger = logging.getLogger(__name__)
tqdm = get_tqdm()


__all__ = [
    'qbip',
    'invasion',
    "capillary_transform",
    "find_trapped_regions2",
    "fill_trapped_voxels",
    "bond_number",
]


def qbip(
    im: npt.NDArray,
    pc: npt.NDArray,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    maxiter: int = None,
    return_sizes: bool = False,
    return_pressures: bool = False,
    conn: str = 'min',
    max_size: int = 0,
):
    r"""
    Performs invasion percolation using a priority queue, optionally including
    the effect of gravity

    The queue-based approach is much faster than the original image-based
    approach [1]_.

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with ``True`` values indicating
        the void space
    pc : ndarray
        Precomputed capillary pressure transform which is used to determine
        the invadability of each voxel, in Pa.
    inlets : ndarray, optional
        A boolean image with ``True`` values indicating the inlet locations.
        If not provided then the beginning of the x-axis is assumed.
    outlets : ndarray, options
        A boolean image with ``True`` values indicating the oulets locations.
        If this is provided then trapped voxels of wetting phase are found and
        all the output images are adjusted accordingly.
    return_sizes : bool
        If `True` then an array containing the size of the sphere which first
        overlapped each pixel is returned. This array is not computed by default
        as computing it increases computation time.
    return_pressures : bool
        If `True` then an array containing the capillary pressure at which
        each pixels was first invaded is returned. This array is not computed by
        default as computing it increases computation time.
    maxiter : int
        The maximum number of iteration to perform.  The default is equal to the
        number of void pixels `im`.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_seq     A numpy array with each voxel value containing the step at
                   which it was invaded.  Uninvaded voxels are set to -1.
        im_satn    A numpy array with each voxel value indicating the saturation
                   present in the domain it was invaded. Solids are given 0, and
                   uninvaded regions are given -1.
        im_pc      If `return_pressures` was set to `True`, then a numpy array with
                   each voxel value indicating the capillary pressure at which it
                   was invaded. Uninvaded voxels have value of ``np.inf``.
        im_size    If `return_sizes` was set to `True`, then a numpy array with
                   each voxel containing the radius of the sphere, in voxels, that
                   first overlapped it.
        ========== =================================================================

    Notes
    -----
    This function operates differently than the original ``ibip``.  Here a
    priority queue (via the `heapq` module from the standard libary) is used to
    maintain an up-to-date list of which voxels should be invaded next.  This
    is much faster than the original approach.

    References
    ----------
    .. [1] Gostick JT, Misaghian N, Yang J, Boek ES. *Simulating volume-controlled
       invasion of a non-wetting fluid in volumetric images using basic image
       processing tools*. `Computers and the Geosciences
       <https://doi.org/10.1016/j.cageo.2021.104978>`_. 158(1), 104978 (2022)

    """
    if maxiter is None:  # Compute numpy of pixels in image
        maxiter = (im == 1).sum()

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if dt is None:
        dt = edt(im)

    dt = np.atleast_3d(dt)
    inlets = np.atleast_3d(inlets)
    im = np.atleast_3d(im == 1)
    pc = np.atleast_3d(pc)

    # Initialize arrays and do some preprocessing
    inv_seq = np.zeros_like(im, dtype=int)
    inv_pc = np.zeros_like(im, dtype=float)
    if return_pressures is False:
        inv_pc *= -np.inf
    inv_size = np.zeros_like(im, dtype=float)
    if return_sizes is False:
        inv_size *= -np.inf

    # Call numba'd inner loop
    sequence, pressure, size = _qbip_inner_loop(
        im=im,
        inlets=inlets,
        dt=dt,
        pc=pc,
        seq=inv_seq,
        pressure=inv_pc,
        size=inv_size,
        maxiter=maxiter,
        conn=conn,
    )
    # Reduce back to 2D if necessary
    sequence = sequence.squeeze()
    pressure = pressure.squeeze()
    size = size.squeeze()
    pc = pc.squeeze()
    im = im.squeeze()

    # Convert invasion image so that uninvaded voxels are set to -1 and solid to 0
    sequence[sequence == 0] = -1
    sequence[~im] = 0
    sequence = make_contiguous(im=sequence, mode='symmetric')
    # Deal with invasion pressures and sizes similarly
    if return_pressures:
        pressure[sequence < 0] = np.inf
        pressure[~im] = 0
    if return_sizes:
        size[sequence < 0] = np.inf
        size[~im] = 0

    if outlets is not None:
        logger.info('Computing trapping and adjusting outputs')
        sequence = find_trapped_regions2(
            seq=sequence,
            im=im,
            outlets=outlets,
            return_mask=False,
            conn=conn,
            max_size=max_size,
        )
        trapped = (sequence == -1).squeeze()
        pressure = pressure.astype(float).squeeze()
        pressure[trapped] = np.inf
        size = size.astype(float)
        size[trapped] = np.inf

    # Create results object for collected returned values
    results = Results()
    results.im_seq = sequence
    results.im_satn = seq_to_satn(sequence)  # convert sequence to saturation
    if return_pressures:
        results.im_pc = pressure
    if return_sizes:
        results.im_size = size
    return results


@njit
def _qbip_inner_loop(
    im,
    inlets,
    dt,
    pc,
    seq,
    pressure,
    size,
    maxiter,
    conn,
):  # pragma: no cover
    # Initialize the heap
    inds = np.where(inlets*im)
    bd = []
    for row, (i, j, k) in enumerate(zip(inds[0], inds[1], inds[2])):
        bd.append([pc[i, j, k], dt[i, j, k], i, j, k])
    hq.heapify(bd)
    # Note which sites have been added to heap already
    processed = inlets*im + ~im  # Add solid phase to be safe
    step = 1  # Total step number
    for _ in range(1, maxiter):
        if len(bd) == 0:
            print(f"Exiting after {step} steps")
            break
        pts = [hq.heappop(bd)]  # Put next site into pts list
        while len(bd) and (bd[0][0] == pts[0][0]):  # Pop any items with equal Pc
            pts.append(hq.heappop(bd))
        for pt in pts:
            # Insert discs of invading fluid into images
            seq = _insert_disk_at_point(
                im=seq,
                i=pt[2], j=pt[3], k=pt[4],
                r=int(pt[1]), v=step, overwrite=False)
            # Putting -inf in images is a numba compatible flag for 'skip'
            if pressure[0, 0, 0] > -np.inf:
                pressure = _insert_disk_at_point(
                    im=pressure,
                    i=pt[2], j=pt[3], k=pt[4],
                    r=int(pt[1]), v=pt[0], overwrite=False)
            if size[0, 0, 0] > -np.inf:
                size = _insert_disk_at_point(
                    im=size, i=pt[2], j=pt[3], k=pt[4],
                    r=int(pt[1]), v=pt[1], overwrite=False)
            # Add neighboring points to heap and processed array
            neighbors = _find_valid_neighbors(
                i=pt[2], j=pt[3], k=pt[4], im=processed, conn=conn)
            for n in neighbors:
                hq.heappush(bd, [pc[n], dt[n], n[0], n[1], n[2]])
                processed[n[0], n[1], n[2]] = True
        step += 1
    return seq, pressure, size


@njit
def _find_valid_neighbors(
    i,
    j,
    im,
    k=0,
    conn='min',
    valid=False
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        if conn == 'min':
            mask = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        elif conn == 'max':
            mask = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        neighbors = []
        for a, x in enumerate(range(i-1, i+2)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-1, j+2)):
                    if (y >= 0) and (y < ylim):
                        if mask[a][b] == 1:
                            if im[x, y] == valid:
                                neighbors.append((x, y))
    else:
        xlim, ylim, zlim = im.shape
        if conn == 'min':
            mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        elif conn == 'max':
            mask = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
        neighbors = []
        for a, x in enumerate(range(i-1, i+2)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-1, j+2)):
                    if (y >= 0) and (y < ylim):
                        for c, z in enumerate(range(k-1, k+2)):
                            if (z >= 0) and (z < zlim):
                                if mask[a][b][c] == 1:
                                    if im[x, y, z] == valid:
                                        neighbors.append((x, y, z))
    return neighbors


@njit
def _insert_disk_at_point(im, i, j, r, v, k=0, overwrite=False):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    i, j, k: int
        The center point of each sphere/disk.  If the image is 2D then `k` can be
        omitted.
    r : array_like
        The radius of the sphere/disk to insert
    v : scalar
        The value to insert
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.
    smooth : boolean
        If `True` (default) then the small bumps on the outer perimeter of each
        face is not present.

    """
    if im.ndim == 2:
        xlim, ylim = im.shape
        for a, x in enumerate(range(i-r, i+r+1)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-r, j+r+1)):
                    if (y >= 0) and (y < ylim):
                        R = ((a - r)**2 + (b - r)**2)**0.5
                        if R < r:
                            if overwrite or (im[x, y] == 0):
                                im[x, y] = v
    else:
        xlim, ylim, zlim = im.shape
        for a, x in enumerate(range(i-r, i+r+1)):
            if (x >= 0) and (x < xlim):
                for b, y in enumerate(range(j-r, j+r+1)):
                    if (y >= 0) and (y < ylim):
                        if zlim > 1:  # For a truly 3D image
                            for c, z in enumerate(range(k-1, k+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if R < r:
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
                        else:  # For 3D image with singleton 3rd dimension
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if R < r:
                                if overwrite or (im[x, y, 0] == 0):
                                    im[x, y, 0] = v
    return im


@njit
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


def invasion(
    im,
    pc,
    dt=None,
    inlets=None,
    outlets=None,
    maxiter=None,
    return_sizes=False,
    return_pressures=False,
    conn='min',
):
    results = qbip(
        im=im,
        pc=pc,
        dt=dt,
        inlets=inlets,
        outlets=outlets,
        maxiter=maxiter,
        return_sizes=return_sizes,
        return_pressures=return_pressures,
        conn=conn,
    )
    return results


invasion.__doc__ = qbip.__doc__


def capillary_transform(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sigma: float = 0.01,
    theta: float = 180,
    g: float = 9.81,
    rho_wp: float = 0,
    rho_nwp: float = 0,
    voxelsize: float = 1e-6,
    spacing: float = None,
):
    r"""
    Uses the Washburn equation to convert distance transform values to capillary
    space

    Parameters
    ----------
    im : ndarray
        A boolean image describing the porous medium with ``True`` values indicating
        the phase of interest.
    dt : ndarray
        The distance transform of the void phase. If not provided it will be
        calculated, so some time can be save if a pre-computed array is already
        available.
    sigma : scalar
        The surface tension of the fluid-fluid interface.
    theta : scalar
        The contact angle of the fluid-fluid-solid system, in degrees.
    g : scalar
        The gravitational constant acting on the fluids. Gravity is assumed to act
        toward the x=0 axis.  To have gravity act in different directions just
        use `np.swapaxes(im, 0, ax)` where `ax` is the desired direction. If gravity
        is not acting directly along one of the principle axes, then use the
        component that is.
    delta_rho : scalar
        The density difference between the fluids.
    voxelsize : scalar
        The resolution of the image
    spacing : scalar
        If a 2D image is provided, this value is used to compute the second
        radii of curvature.  Setting it to `inf` will make the calculation truly
        2D since only one radii of curvature is considered. Setting it to `None`
        will force the calculation to be 3D.  If `im` is 3D this argument is
        ignored.

    Notes
    -----
    All physical properties should be in self-consistent units, and it is strongly
    recommended to use SI for everything.

    """
    delta_rho = rho_nwp - rho_wp
    if dt is None:
        dt = edt(im)
    if (im.ndim == 2) and (spacing is not None):
        pc = -sigma*np.cos(np.deg2rad(theta))*(1/(dt*voxelsize) + 2/spacing)
    else:
        pc = -2*sigma*np.cos(np.deg2rad(theta))/(dt*voxelsize)
    if delta_rho > 0:
        h = ramp(im.shape, inlet=0, outlet=im.shape[0], axis=0)*voxelsize
        pc = pc + delta_rho*g*h
    elif delta_rho < 0:
        h = ramp(im.shape, inlet=im.shape[0], outlet=0, axis=0)*voxelsize
        pc = pc + delta_rho*g*h
    return pc


def bond_number(
    im: npt.NDArray,
    delta_rho: float,
    g: float,
    sigma: float,
    voxelsize: float,
    source: str = 'lt',
    method: str = 'median',
    mask: bool = False,
):
    r"""
    Computes the Bond number for an image

    Parameters
    ----------
    im : ndarray
        The image of the domain with `True` values indicating the phase of interest
        space
    delta_rho : float
        The difference in the density of the non-wetting and wetting phase
    g : float
        The gravitational constant for the system
    sigma : float
        The surface tension of the fluid pair
    voxelsize : float
        The size of the voxels
    source : str
        The source of the pore size values to use when computing the characteristic
        length *R*. Options are:

        ============== =============================================================
        Option         Description
        ============== =============================================================
        dt             Uses the distance transform
        lt             Uses the local thickness
        ============== =============================================================

    method : str
        The method to use for finding the characteristic length *R* from the
        values in `source`. Options are:

        ============== =============================================================
        Option         Description
        ============== =============================================================
        mean           The arithmetic mean (using `numpy.mean`)
        min (or amin)  The minimum value (using `numpy.amin`)
        max (or amax)  The maximum value (using `numpy.amax`)
        mode           The mode of the values (using `scipy.stats.mode`)
        gmean          The geometric mean of the values (using `scipy.stats.gmean`)
        hmean          The harmonic mean of the values (using `scipy.stats.hmean`)
        pmean          The power mean of the values (using `scipy.stats.pmean`)
        ============== =============================================================

    mask : bool
        If `True` then the distance values in `source` are masked by the skeleton
        before computing the average value using the specified `method`.
    """
    if mask is True:
        mask = skeletonize_3d(im)
    else:
        mask = im

    if source == 'dt':
        dvals = edt(im)
    elif source == 'lt':
        dvals = local_thickness(im)
    else:
        raise Exception(f"Unrecognized source {source}")

    if method in ['median', 'mean', 'amin', 'amax']:
        f = getattr(np, method)
    elif method in ['min', 'max']:
        f = getattr(np, 'a' + method)
    elif method in ['pmean', 'hmean', 'gmean', 'mode']:
        f = getattr(spst, method)
    else:
        raise Exception(f"Unrecognized method {method}")
    R = f(dvals[mask])
    Bo = abs(delta_rho*g*(R*voxelsize)**2/sigma)
    return Bo


def fill_trapped_voxels(
    seq: npt.NDArray,
    trapped: npt.NDArray = None,
    max_size: int = 1,
    conn: str = 'min',
):
    r"""
    Finds small isolated clusters of voxels which were identified as trapped and
    sets them to invaded.

    Parameters
    ----------
    seq : ndarray
        The sequence map resulting from an invasion process where trapping has
        been applied, such that trapped voxels are labelled -1.
    trapped : ndarray, optional
        The boolean array of the trapped voxels. If this is not available than all
        voxels in `seq` with a value < 0 are used.
    max_size : int
        The maximum size of the clusters which are to be filled. Clusters larger
        than this size are left trapped.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels when looking for sequence values to place into un-trapped voxels.
        Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    Notes
    -----
    This function has to essentially guess which sequence value to put into each
    un-trapped voxel so the sequence values can differ between the output of
    this function and the result returned by the various invasion algorithms where
    the trapping is computed internally. However, the fluid configuration for a
    given saturation will be nearly identical.

    """
    if trapped is None:
        trapped = seq < 0

    strel = ps_round(r=1, ndim=seq.ndim, smooth=False)
    size = region_size(trapped, strel=strel)
    mask = (size <= max_size)*(size > 0)
    trapped[mask] = False

    if conn == 'min':
        strel = ps_round(r=1, ndim=seq.ndim, smooth=False)
    else:
        strel = ps_rect(w=3, ndim=seq.ndim)
    mx = maximum_filter(seq*~trapped, footprint=strel)
    mx = flood_func(mx, np.amax, labels=label(mask, structure=strel)[0])
    seq[mask] = mx[mask]

    results = Results()
    results.im_seq = seq
    results.im_trapped = trapped
    return results


def find_trapped_regions2(
    seq: npt.NDArray,
    im: npt.NDArray,
    outlets: npt.NDArray,
    return_mask: bool = True,
    conn: str = 'min',
    max_size: int = 0,
    mode: str = 'queue',
):
    r"""
    Finds clusters of trapped voxels given a set of outlets

    Parameters
    ----------
    seq : ndarray
        The image containing the invasion sequence values from the drainage or
        invasion simulation.
    im : ndarray
        The boolean image of the porous material with `True` indicating the phase
        of interest.
    outlets : ndarray
        A boolean image the same shape as `seq` with `True` values indicating the
        outlet locations for the defending phase.
    return_mask : boolean
        If `True` (default) a boolean image is returned with `True` values indicating
        which voxels were trapped. If `False`, then an image containing updated
        invasion sequence values is returned, with -1 indicating the trapped
        voxels.
    mode : str
        Controls the method used to find trapped voxels. Options are:

        ========== =================================================================
        Option     Description
        ========== =================================================================
        'cluster'  This method finds clusters of disconnected voxels with an
                   invasion sequence less than or equal to the values on the outlet.
                   It works well for invasion sequence maps which were produced by
                   pressure-based simulations (IBOP).
        'queue'    This method uses a queue-based method which is much faster if
                   the invasion was performed using IBIP or QBIP, but can end up
                   being slower than `'cluster'` if IBOP was used.
        ========== =================================================================

    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    max_size : int
        Any cluster of trapped voxels smaller than this size will be set to *not
        trapped*. This is useful to remove small voxels along edges of the void
        space, which appear to be trapped due to the jagged nature of the digital
        image. The default is 0, meaning this adjustment is not applied, but a
        value of 3 or 4 is probably suitable to activate this adjustment.

    Returns
    -------
    Depending on the value of `return_mask` this function either returns a
    boolean image with `True` value indicating trapped voxels, or a image
    containing updated invasion sequence values with trapped voxels given a
    value of -1.

    Notes
    -----
    This currently only works if the image has been completely filled to the outlets
    so needs to get the special case treatment that I added to the original
    function.
    """
    im = im > 0
    if mode == 'queue':
        # Make sure outlets are masked correctly and convert to 3d
        out_temp = np.atleast_3d(outlets*(seq > 0))
        # Initialize im_trapped array
        im_trapped = np.ones_like(out_temp, dtype=bool)
        # Convert seq to negative numbers and convert to 3d
        seq_temp = np.atleast_3d(-1*seq)
        # Note which sites have been added to heap already
        edge = out_temp*np.atleast_3d(im) + np.atleast_3d(~im)
        # seq = np.copy(np.atleast_3d(seq))
        trapped = _trapped_regions_inner_loop(
            seq=seq_temp,
            edge=edge,
            trapped=im_trapped,
            outlets=out_temp,
            conn=conn,
        )

        if return_mask:
            trapped = trapped.squeeze()
            trapped[~im] = 0
            return trapped
        else:
            if max_size > 0:  # Fix pixels on solid surfaces
                seq, trapped = fill_trapped_voxels(seq_temp, max_size=10)
            seq = np.squeeze(seq)
            trapped = np.squeeze(trapped)
            seq[trapped] = -1
            seq[~im] = 0
            seq = make_contiguous(im=seq, mode='symmetric')
            return seq
    else:
        raise NotImplementedError("Sorry, cluster is not implemented yet")


@njit
def _trapped_regions_inner_loop(
    seq,
    edge,
    trapped,
    outlets,
    conn,
):  # pragma: no cover
    # Initialize the binary heap
    inds = np.where(outlets)
    bd = []
    for row, (i, j, k) in enumerate(zip(inds[0], inds[1], inds[2])):
        bd.append([seq[i, j, k], i, j, k])
    hq.heapify(bd)
    minseq = np.amin(seq)
    step = 1
    maxiter = np.sum(seq < 0)
    for _ in range(1, maxiter):
        if len(bd):  # Put next site into pts list
            pts = [hq.heappop(bd)]
        else:
            print(f"Exiting after {step} steps")
            break
        # Also pop any other points in list with same value
        while len(bd) and (bd[0][0] == pts[0][0]):
            pts.append(hq.heappop(bd))
        while len(pts):
            pt = pts.pop()
            if (pt[0] >= minseq) and (pt[0] < 0):
                trapped[pt[1], pt[2], pt[3]] = False
                minseq = pt[0]
            # Add neighboring points to heap and edge
            neighbors = \
                _find_valid_neighbors(i=pt[1], j=pt[2], k=pt[3], im=edge, conn=conn)
            for n in neighbors:
                hq.heappush(bd, [seq[n], n[0], n[1], n[2]])
                edge[n[0], n[1], n[2]] = True
        # if step % 1000 == 0:
        #     print(f'completed {str(step)} steps')
        step += 1
    return trapped


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy

    # %%
    im = ~ps.generators.random_spheres([400, 200], r=20, seed=0, clearance=10)

    inlets = np.zeros_like(im)
    inlets[0, :] = True
    inlets = inlets*im
    pc = capillary_transform(im)
    ip = qbip(im, pc=pc, inlets=inlets)

    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    outlets = outlets*im
    ps.tools.tic()
    trapped_new = find_trapped_regions2(
        seq=ip.im_seq, im=im, outlets=outlets, return_mask=False)
    ps.tools.toc()
    ps.tools.tic()
    trapped = ps.filters.find_trapped_regions(
        seq=ip.im_seq, outlets=outlets, bins=None, return_mask=False)
    ps.tools.toc()

    # %%
    cm = copy(plt.cm.turbo)
    cm.set_under('grey')
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(
        ip.im_seq/im,
        origin='lower',
        interpolation='none',
        vmin=0.0001,
        cmap=cm,
    )
    ax[1].imshow(
        trapped/im,
        origin='lower',
        interpolation='none',
        vmin=0.0001,
        cmap=cm,
    )
    ax[2].imshow(
        trapped_new/im,
        origin='lower',
        interpolation='none',
        vmin=0.0001,
        cmap=cm,
    )
