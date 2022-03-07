import numpy as np
from edt import edt
from porespy.tools import get_tqdm
import scipy.ndimage as spim
from porespy.filters import trim_disconnected_blobs, seq_to_satn
from porespy.tools import get_border, make_contiguous
from porespy.tools import Results
import numba
from porespy import settings
import heapq as hq
from random import seed
tqdm = get_tqdm()


__all__ = [
    'ibip_w_gravity',
    'invasion',
    ]


import heapq as hq
def invasion(im, voxel_size, inlets=None, dt=None, pc=None, sigma=0.072, theta=180, delta_rho=998, g=0, maxiter=10000):
    r"""
    Perform image-based invasion percolation in the presence of gravity

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
                   capillary pressure at which it was invaded. In invaded
                   voxels have value of ``np.inf``.
        im_satn    A numpy array with each voxel value indicating the global
                   saturation value at the point it was invaded
        pc         1D array of capillary pressure values that were applied
        swnp       1D array of non-wetting phase saturations for each applied
                   value of capillary pressure (``pc``).
        ========== ============================================================
    """

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if dt is None:
        dt = edt(im)

    if pc is None:
        pc = -2*sigma*np.cos(np.deg2rad(theta))/(dt*voxel_size)
        if (g * delta_rho) != 0:
            h = np.ones_like(im)
            h[0, ...] = False
            h = edt(h) * voxel_size
            pc = pc + delta_rho * g * h

    pc = np.around(pc, decimals=0)  # Convert the dt to nearest integer
    dt = dt.astype(int)  # Convert the dt to nearest integer

    # Prepare heap
    inds = np.where(inlets*im)
    bd = list(zip(pc[inds], dt[inds], *inds))
    hq.heapify(bd)
    # Note which sites have been added to heap already
    edge = np.copy(inlets)

    # Initial arrays
    inv = np.zeros_like(im, dtype=int)
    pressures = np.zeros_like(im, dtype=float)

    for step in tqdm(range(1, maxiter), **settings.tqdm):
        # Find sites to add spheres, which may be multiple
        try:  # pop the top item from heap to get current size
            pts = [hq.heappop(bd)]
        except IndexError:  # bd is empty, exit
            print(f'Exiting after {step} steps')
            break
        try:  # pop any additional items with the same size
            while bd[0][0] == pts[0][0]:
                pts.append(hq.heappop(bd))
        except IndexError:  # if bd gets emptied, stop popping and finish loop
            pass
        # Add spheres to image
        p = pts[0][0]
        r = pts[0][1]
        coords = np.vstack([pts[i][2:] for i in range(len(pts))]).T
        inv = _insert_disks_at_points(im=inv, coords=coords,
                                      r=r, v=step, smooth=True)
        pressures = _insert_disks_at_points(im=pressures, coords=coords,
                                            r=r, v=p, smooth=True)
        # Check neighborhood around coords and add new points if any
        for pt in pts:
            # Get extended slices around each point in pts
            x, y = find_neighbor_coordinates(pt=pt[2:], shape=im.shape)
            for item in zip(x, y):
                if (edge[item] == False):
                    hq.heappush(bd, (pc[item], dt[item], *item))
                    edge[item] = True

    # Convert inv image so that uninvaded voxels are set to -1 and solid to 0
    temp = inv == 0  # Uninvaded voxels are set to -1 after _ibip
    inv[temp] = -1
    inv[~im] = 0
    inv = make_contiguous(im=inv, mode='symmetric')
    # Deal with invasion sizes similarly
    temp = pressures == 0
    # pressures[temp] = np.inf
    pressures[~im] = 0

    # Create results object for collected returned values
    results = Results()
    results.im_seq = inv
    results.im_pc = pressures
    results.im_satn = seq_to_satn(inv)  # convert sequence to saturation
    return results


_x = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int)
_y = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1], dtype=int)
_conn = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int).flatten()


def find_neighbor_coordinates(pt, shape):
    xi = _x + pt[0]
    yi = _y + pt[1]
    mask_x = (xi >= 0) * (xi < shape[0])
    mask_y = (yi >= 0) * (yi < shape[1])
    mask = np.array(mask_x * mask_y * _conn, dtype=bool)
    return xi[mask], yi[mask]


def ibip_w_gravity(im, inlets=None, dt=None, maxiter=10000, g=0):
    r"""
    Performs invasion percolation on given image using iterative image dilation

    Parameters
    ----------
    im : ND-array
        Boolean array with ``True`` values indicating void voxels
    inlets : ND-array
        Boolean array with ``True`` values indicating where the invading fluid
        is injected from.  If ``None``, all faces will be used.
    dt : ND-array (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    maxiter : scalar
        The number of steps to apply before stopping.  The default is to run
        for 10,000 steps which is almost certain to reach completion if the
        image is smaller than about 250-cubed.

    Returns
    -------
    results : Results object
        A custom object with the following two arrays as attributes:

        'inv_sequence'
            An ndarray the same shape as ``im`` with each voxel labelled by
            the sequence at which it was invaded.

        'inv_size'
            An ndarray the same shape as ``im`` with each voxel labelled by
            the ``inv_size`` at which was filled.

    See Also
    --------
    porosimetry

    """
    # Process the boundary image
    if inlets is None:
        inlets = get_border(shape=im.shape, mode='faces')
    bd = np.copy(inlets > 0)
    if dt is None:  # Find dt if not given
        dt = edt(im)

    # This pc calc is still very basic, just proof-of-concept for now
    pc = 1/(dt*1e-4)
    h = np.ones_like(im)
    h[0, ...] = False
    h = edt(h)
    pc = pc + g * h
    pc = pc.astype(int)  # Convert the pc to nearest integer
    dt = dt.astype(int)  # Convert the dt to nearest integer

    # Initialize inv image with -1 in the solid, and 0's in the void
    inv = -1*(~im)
    pressures = -1*(~im)
    scratch = np.copy(bd)
    for step in tqdm(range(1, maxiter), **settings.tqdm):
        pt = _where(bd)
        scratch = np.copy(bd)
        temp = _insert_disks_at_points(im=scratch, coords=pt,
                                       r=1, v=1, smooth=False)
        # Reduce to only the 'new' boundary
        edge = temp*(pc > 0)
        if ~edge.any():
            print('No edges found, exiting')
            break
        # Find the minimum value of the pc map underlaying the new edge
        pc_min = pc[edge].min()
        # Find all values of the dt with that pressure
        pc_thresh = (pc <= pc_min)*im
        # Extract the actual coordinates of the insertion sites
        pt = _where(edge*pc_thresh)
        r = dt[tuple([i[0] for i in pt])]
        inv = _insert_disks_at_points(im=inv, coords=pt,
                                      r=r, v=step, smooth=True)
        pressures = _insert_disks_at_points(im=pressures, coords=pt,
                                            r=r, v=pc_min, smooth=True)
        pc, bd = _update_pc_and_bd(pc, bd, pt)
    # Convert inv image so that uninvaded voxels are set to -1 and solid to 0
    temp = inv == 0  # Uninvaded voxels are set to -1 after _ibip
    inv[~im] = 0
    inv[temp] = -1
    inv = make_contiguous(im=inv, mode='symmetric')
    # Deal with invasion sizes similarly
    temp = pressures == 0
    pressures[~im] = 0
    pressures[temp] = -1
    results = Results()
    results.inv_sequence = inv
    results.inv_pressures = pressures
    return results


@numba.jit(nopython=True, parallel=False)
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


@numba.jit(nopython=True)
def _update_pc_and_bd(pc, bd, pt):
    if pc.ndim == 2:
        for i in range(pt.shape[1]):
            bd[pt[0, i], pt[1, i]] = True
            pc[pt[0, i], pt[1, i]] = 0
    else:
        for i in range(pt.shape[1]):
            bd[pt[0, i], pt[1, i], pt[2, i]] = True
            pc[pt[0, i], pt[1, i], pt[2, i]] = 0
    return pc, bd


@numba.jit(nopython=True)
def _make_disks(r, smooth=True):  # pragma: no cover
    r"""
    Returns a list of disks from size 0 to ``r``

    Parameters
    ----------
    r : int
        The size of the largest disk to generate
    smooth : bool
        Indicates whether the disks should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    disks : list of ND-arrays
        A list containing the disk images, with the disk of radius R at index
        R of the list, meaning it can be accessed as ``disks[R]``.

    """
    disks = []
    for val in range(0, r+1):
        disk = _make_disk(val, smooth)
        disks.append(disk)
    return disks


@numba.jit(nopython=True, parallel=False)
def _make_balls(r, smooth=True):  # pragma: no cover
    r"""
    Returns a list of balls from size 0 to ``r``

    Parameters
    ----------
    r : int
        The size of the largest ball to generate
    smooth : bool
        Indicates whether the balls should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    balls : list of ND-arrays
        A list containing the ball images, with the ball of radius R at index
        R of the list, meaning it can be accessed as ``balls[R]``.

    """
    balls = [np.atleast_3d(np.array([]))]
    for val in range(1, r):
        ball = _make_ball(val, smooth)
        balls.append(ball)
    return balls


def insert_disk(im, c, s, v=1):
    r = int(s.shape[0]/2)
    i_im = tuple([slice(c[i] - r - 1 if (c[i] - r - 1) > 0 else 0,
                        min(c[i] + r, im.shape[i]-1))
                  for i in range(im.ndim)])
    i_s = tuple([slice(0 if (c[i] - r - 1) > 0 else (r - c[i] + 1),
                       s.shape[i] if (c[i] + r) < im.shape[i]
                       else (s.shape[i] - (c[i] + r - im.shape[i])) - 1)
                 for i in range(im.ndim)])
    im[i_im] += s[i_s]*(im[i_im] == 0)*v
    return im


@numba.jit(nopython=True, parallel=False)
def _insert_disks_at_points(im, coords, r, v, s=None, smooth=True):  # pragma: no cover
    r"""
    Insert spheres (or disks) into the given ND-image at given locations

    This function uses numba to accelerate the process, and does not
    overwrite any existing values (i.e. only writes to locations containing
    zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    r : int
        The radius of all the spheres/disks to add. It is assumed that they
        are all the same radius.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.

    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        if s is None:
            s = _make_disk(r, smooth)
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            if (s[a, b] == 1) and (im[x, y] == 0):
                                im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        if s is None:
            s = _make_ball(r, smooth)
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    if (s[a, b, c] == 1) and (im[x, y, z] == 0):
                                        im[x, y, z] = v
    return im


@numba.jit(nopython=True, parallel=False)
def _make_disk(r, smooth=True):  # pragma: no cover
    s = np.zeros((2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            if ((i - r)**2 + (j - r)**2)**0.5 <= thresh:
                s[i, j] = 1
    return s


@numba.jit(nopython=True, parallel=False)
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


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    ps.settings.tqdm['leave'] = True
    np.random.seed(0)
    im = ps.generators.blobs([400, 400], porosity=0.7, blobiness=1.5)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    # fig, ax = plt.subplots(2, 2)

    # inv_g, pc_g = ibip_w_gravity(im=im, inlets=inlets, g=9.81, maxiter=50000)
    # satn_g = ps.filters.seq_to_satn(inv_g, im=im)

    ip = invasion(im=im, inlets=inlets, voxel_size=1e-5, g=0, maxiter=10000)
    satn_g = ps.filters.seq_to_satn(ip.im_seq, im=im)
#     # ax[0][0].imshow(inv_g/im, origin='lower', interpolation='none')
#     # ax[0][1].imshow(pc_g/im, origin='lower', interpolation='none')

#     ani = ps.visualization.satn_to_movie(im=im, satn=satn_g)
#     # ani.save('image_based_ip_w_gravity.gif', writer='imagemagick', fps=10)

#     # inv, pc = ibip_w_gravity(im=im, inlets=inlets, g=0)
#     # ax[1][0].imshow(inv/im, origin='lower', interpolation='none')
#     # ax[1][1].imshow(pc/im, origin='lower', interpolation='none')

#     # satn = ps.filters.seq_to_satn(inv, im=im)
#     # ani = ps.visualization.satn_to_movie(im=im, satn=satn)
#     # ani.save('image_based_ip.gif', writer='imagemagick', fps=3)
