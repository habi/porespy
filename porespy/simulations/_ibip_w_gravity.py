import heapq as hq
import numpy as np
from edt import edt
from numba import njit
from porespy.filters import seq_to_satn
from porespy.tools import (
    get_tqdm,
    make_contiguous,
    Results,
)


tqdm = get_tqdm()


__all__ = [
    'invasion',
]


def invasion(
    im,
    pc,
    inlets=None,
    maxiter=None,
    return_sizes=False,
    return_pressures=False,
):
    r"""
    Perform image-based invasion percolation in the presence of gravity

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with ``True`` values indicating
        the void space
    pc : ndarray
        Precomputed capillary pressure values which are used to determine
        the invadability of each voxel, in Pa.
    inlets : ndarray, optional
        A boolean image with ``True`` values indicating the inlet locations.
        If not provided then the beginning of the x-axis is assumed.
    return_sizes : bool
        If `True` then array containing the size of the sphere which first
        overlapped each pixel is returned. This array is not computed by default
        so requesting it increases computation time.
    return_pressures : bool
        If `True` then array containing the capillary pressure at which
        each pixels was first invaded is returned. This array is not computed by
        default so requesting it increases computation time.
    maxiter : int
        The maximum number of iteration to perform.  The default is equal to the
        number of void pixels `im`.

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
    binary heap (via the `heapq` module from the standard libary) is used to
    maintain an up-to-date list of which voxels should be invaded next.  This
    is much faster than the original approach.

    """
    if maxiter is None:
        maxiter = int(np.prod(im.shape)*(im.sum()/im.size))

    im = np.atleast_3d(im)
    pc = np.atleast_3d(pc)

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
    else:
        inlets = np.atleast_3d(inlets)

    dt = edt(im)
    dt = np.around(dt, decimals=0).astype(int)

    pc = np.around(pc, decimals=0)

    # Initialize arrays and do some preprocessing
    inv_seq = np.zeros_like(im, dtype=int)
    inv_pc = np.zeros_like(im, dtype=float)
    if return_pressures is False:
        inv_pc *= -np.inf
    inv_size = np.zeros_like(im, dtype=float)
    if return_sizes is False:
        inv_size *= -np.inf

    # Call numba'd inner loop
    sequence, pressure, size = _ibip_inner_loop(
        im=im,
        inlets=inlets,
        dt=dt,
        pc=pc,
        seq=inv_seq,
        pressure=inv_pc,
        size=inv_size,
        maxiter=maxiter,
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
def _ibip_inner_loop(
    im,
    inlets,
    dt,
    pc,
    seq,
    pressure,
    size,
    maxiter,
):  # pragma: no cover
    step = 1
    # Initialize the binary heap
    inds = np.where(inlets*im)
    bd = []
    for row, (i, j, k) in enumerate(zip(inds[0], inds[1], inds[2])):
        bd.append([pc[i, j, k], dt[i, j, k], i, j, k])
    hq.heapify(bd)
    # Note which sites have been added to heap already
    edge = inlets*im + ~im
    delta_step = 0
    for _ in range(1, maxiter):
        if len(bd):  # Put next site into pts list
            pts = [hq.heappop(bd)]
        else:
            print(f"Exiting after {step} steps")
            break
        while len(bd) and (bd[0][0] == pts[0][0]):
            pts.append(hq.heappop(bd))
        for pt in pts:
            # Insert discs of invading fluid into images
            seq = _insert_disk_at_point(im=seq, i=pt[2], j=pt[3], k=pt[4], r=pt[1],
                                        v=step, overwrite=False)
            if pressure[0, 0, 0] > -np.inf:
                pressure = _insert_disk_at_point(im=pressure, i=pt[2], j=pt[3], k=pt[4],
                                                 r=pt[1], v=pt[0], overwrite=False)
            if size[0, 0, 0] > -np.inf:
                size = _insert_disk_at_point(im=size, i=pt[2], j=pt[3], k=pt[4],
                                             r=pt[1], v=pt[1], overwrite=False)
            # Add neighboring points to heap and edge
            neighbors = _find_valid_neighbors(i=pt[2], j=pt[3], k=pt[4], im=edge, conn=26)
            for n in neighbors:
                hq.heappush(bd, [pc[n], dt[n], n[0], n[1], n[2]])
                edge[n[0], n[1], n[2]] = True
                delta_step = 1
        step += delta_step
        delta_step = 0
    return seq, pressure, size


@njit
def _find_valid_neighbors(i, j, im, k=0, conn=4, valid=False):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        if conn == 4:
            mask = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        else:
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
        if conn == 6:
            mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
        else:
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
