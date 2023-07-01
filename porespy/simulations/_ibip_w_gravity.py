import heapq as hq
import numpy as np
from edt import edt
from numba import njit, prange
from porespy.filters import seq_to_satn
from porespy.tools import (
    get_tqdm,
    make_contiguous,
    Results,
)


tqdm = get_tqdm()


__all__ = [
    'invasion',
    'pc_2D',
    'dt_to_pc',
]


def invasion(
    im,
    voxel_size,
    inlets=None,
    pc=None,
    dt=None,
    sigma=0.072,
    theta=180,
    delta_rho=998,
    g=0,
    maxiter=-1,
):
    r"""
    Perform image-based invasion percolation in the presence of gravity

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with ``True`` values indicating
        the void space
    voxel_size : float
        The length of a voxel side, in meters
    inlets : ndarray, optional
        A boolean image with ``True`` values indicating the inlet locations.
        If not provided then the beginning of the x-axis is assumed.
    pc : ndarray, optional
        Precomputed capillary pressure values which are used to determine
        the invadability of each voxel, in Pa.  If not provided then it is
        computed assuming the Washburn equation using the given values of
        ``theta`` and ``sigma``.
    dt : ndarray, optional
        The distance transform of the void space is necessary to know the size of
        spheres to draw. If not provided it will be computed but some time can be
        saved by providing it if available.
    sigma, theta: float, optional
        The surface tension and contact angle to use when computing ``pc``.
    delta_rho, g : float, optional
        The phase density difference and gravitational constant. If either is
        set to 0 the gravitational effects are neglected.  The default is to
        disable gravity (``g=0``)
    maxiter : int
        The maximum number of iteration to perform.  The default is equal to the
        number of void pixels `im`.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== ============================================================
        Attribute  Description
        ========== ============================================================
        im_seq     A numpy array with each voxel value containing the step at
                   which it was invaded.  Uninvaded voxels are set to -1.
        im_pc      A numpy array with each voxel value indicating the
                   capillary pressure at which it was invaded. Uninvaded
                   voxels have value of ``np.inf``.
        im_sizes   A numpy array with each voxel value indicating the radii
                   of the spheres inserted.
        im_satn    A numpy array with each voxel value indicating the global
                   saturation value at the point it was invaded
        ========== ============================================================

    Notes
    -----
    This function operates differently than the original ``ibip``.  Here a
    binary heap (via the `heapq` module from the standard libary) is used to
    maintain an up-to-date list of which voxels should be invaded next.  This
    is much faster than the original approach which scanned the entire image
    for invasion sites on each step.

    """
    if maxiter < 0:
        maxiter = int(np.prod(im.shape)*(im.sum()/im.size))

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    dt = edt(im)

    if pc is None:
        pc = -2*sigma*np.cos(np.deg2rad(theta))/(dt*voxel_size)
        if (g * delta_rho) != 0:
            h = np.ones_like(im)
            h[0, ...] = False
            h = edt(h) * voxel_size
            pc = pc + delta_rho * g * h

    # Initialize arrays and do some preprocessing
    sequence = np.zeros_like(im, dtype=int)
    pressure = np.zeros_like(im, dtype=float)
    dt = dt.astype(int)
    disks = _make_disks(dt.max()+1, smooth=False)
    sequence, pressure = _ibip_inner_loop(
        im=im,
        inlets=inlets,
        dt=dt,
        pc=pc,
        seq=sequence,
        pressure=pressure,
        maxiter=maxiter,
        disks=disks,
    )
    # Convert invasion image so that uninvaded voxels are set to -1 and solid to 0
    sequence[sequence == 0] = -1
    sequence[~im] = 0
    sequence = make_contiguous(im=sequence, mode='symmetric')
    # Deal with invasion pressures similarly
    pressure[sequence < 0] = np.inf
    pressure[~im] = 0

    # Create results object for collected returned values
    results = Results()
    results.im_seq = sequence
    results.im_sizes = -2*sigma*np.cos(np.deg2rad(theta))/(pressure)
    results.im_pc = pressure
    results.im_satn = seq_to_satn(sequence)  # convert sequence to saturation
    return results


@njit
def _ibip_inner_loop(
    im,
    inlets,
    dt,
    pc,
    seq,
    pressure,
    maxiter,
    disks
):  # pragma: no cover
    # Initialize the binary heap
    inds = np.where(inlets*im)
    bd = []
    for row, (i, j) in enumerate(zip(inds[0], inds[1])):
        bd.append([pc[i, j], dt[i, j], i, j])
    hq.heapify(bd)
    # Note which sites have been added to heap already
    edge = inlets + ~im
    step = 1
    for _ in range(1, maxiter):
        if len(bd):
            pt = hq.heappop(bd)
        else:
            break
        # Insert discs of invadign fluid to images
        seq = _insert_disk_at_point(im=seq, i=pt[2], j=pt[3], r=pt[1],
                                    v=step, disks=disks)
        pressure = _insert_disk_at_point(im=pressure, i=pt[2], j=pt[3], r=pt[1],
                                          v=pt[0], disks=disks)
        # Add neighboring points to heap
        neighbors = _find_valid_neighbors(pt[2], pt[3], edge)
        for n in neighbors:
            hq.heappush(bd, [pc[n], dt[n], n[0], n[1]])
            edge[n] = True
        # Ensures multiple spheres of same size in a row have same step number
        # if len(bd) and (pt[0] < bd[0][0]):
        #     step += 1
        step += 1
    return seq, pressure


@njit
def _find_valid_neighbors(i, j, im, conn=4, valid=False):  # pragma: no cover
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
    return neighbors


@njit
def _insert_disk_at_point(
    im,
    i,
    j,
    r,
    v,
    disks,
    overwrite=False
):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    i, j : int
        The center point of each sphere/disk given
    r : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    disks : ndarray
        An array containing the disk to insert. It is faster to pre-generate these
        and pass in the desired one than the generate it using `r` each time.
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.

    """
    s = disks[int(r)]
    W = s.shape[0]
    lo, hi = int((W-1)/2)-r, int((W-1)/2)+r+1
    s = s[lo:hi, lo:hi]
    xlim, ylim = im.shape
    for a, x in enumerate(range(i-r, i+r+1)):
        if (x >= 0) and (x < xlim):
            for b, y in enumerate(range(j-r, j+r+1)):
                if (y >= 0) and (y < ylim):
                    if s[a, b] == 1:
                        if overwrite or (im[x, y] == 0):
                            im[x, y] = v
    return im


def pc_2D(r, s, sigma, theta):
    pc = -sigma*np.cos(np.radians(theta))*(1/r + 1/s)
    return pc


def dt_to_pc(f, **kwargs):
    pc = f(**kwargs)
    return pc


@njit
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


@njit
def _make_disk(r, smooth=True):  # pragma: no cover
    W = int(2*r+1)
    s = np.zeros((W, W), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(W):
        for j in range(W):
            if ((i - r)**2 + (j - r)**2)**0.5 <= thresh:
                s[i, j] = 1
    return s


@njit
def _make_ball(r, smooth=True):  # pragma: no cover
    s = np.zeros((2*r+1, 2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            for k in range(2*r+1):
                if ((i - r)**2 + (j - r)**2 + (k - r)**2)**0.5 <= thresh:
                    s[i, j, k] = 1
    return s


def _make_disks(r_max, smooth=False):
    W = int(2*r_max + 1)
    sph = np.zeros([int(r_max) + 1, W, W], dtype=bool)
    for r in range(1, int(r_max) + 1):
        lo, hi = int((W-1)/2)-r, int((W-1)/2)+r+1
        sph[r, lo:hi, lo:hi] = _make_disk(r, smooth=smooth)
    return sph


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.ndimage as spim

    # im = ps.generators.blobs([800, 800], porosity=0.7, blobiness=2, seed=2)
    im = ps.generators.overlapping_spheres([800, 800], porosity=0.55, r=20, seed=2)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    ibip_new = ps.simulations.invasion(im=im, inlets=inlets, voxel_size=1e-4)
    # fig, ax = plt.subplots()
    # ax.imshow(ibip_new.im_satn)

    seq = np.copy(ibip_new.im_seq)
    seq_orig = np.copy(ibip_new.im_seq)
    outlets = np.zeros_like(inlets)
    outlets[-1, :] = True
    outlets *= im

# %%
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(seq_orig/im)
    # seq = ps.filters.find_trapped_regions(seq, outlets=outlets, bins=None, return_mask=False)
    ax[1].imshow(seq/im/~outlets)

