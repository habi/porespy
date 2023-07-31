import numpy as np
from edt import edt
from _ibip_w_gravity import invasion, _find_valid_neighbors
from porespy.tools import make_contiguous
import heapq as hq
from numba import njit


def capillary_transform(
    im,
    dt=None,
    sigma=0.01,
    theta=180,
    g=9.81,
    rho=0,
    voxelsize=1e-6,
    spacing=100e-6
):
    r"""
    Uses the Washburn equation to convert distance transform values to capillary
    space

    Parameters
    ----------
    im : ndarray
        A boolean image describing the porous medium with ``True`` values indicating
        the phase of interest.
    """
    if dt is None:
        dt = edt(im)
    if im.ndim == 2:
        pc = -sigma*np.cos(np.deg2rad(theta))*(1/(dt*voxelsize) + 1/spacing)
    else:
        pc = -2*sigma*np.cos(np.deg2rad(theta))/(dt*voxelsize)
    return pc


def find_trapped_regions(seq, outlets, bins=None, return_mask=True):
    r"""
    """
    # Make sure outlets are masked correctly and convert to 3d
    out_temp = np.atleast_3d(outlets*(seq > 0))
    # Initialize im_trapped array
    im_trapped = np.ones_like(out_temp, dtype=bool)
    # Convert seq to negative numbers and convert ot 3d
    seq_temp = np.atleast_3d(-1*seq)
    trapped = _ibip_inner_loop(seq=seq_temp, trapped=im_trapped, outlets=out_temp)
    trapped = trapped.squeeze()
    if return_mask:
        return trapped
    else:
        seq = np.copy(seq)
        seq[trapped] = -1
        seq = make_contiguous(im=seq, mode='symmetric')
        return seq


@njit
def _ibip_inner_loop(seq, trapped, outlets):  # pragma: no cover
    # Initialize the binary heap
    inds = np.where(outlets)
    bd = []
    for row, (i, j, k) in enumerate(zip(inds[0], inds[1], inds[2])):
        bd.append([seq[i, j, k], i, j, k])
    hq.heapify(bd)
    # Note which sites have been added to heap already
    edge = outlets*np.atleast_3d(im) + np.atleast_3d(~im)
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
            neighbors = _find_valid_neighbors(i=pt[1], j=pt[2], k=pt[3], im=edge, conn=26)
            for n in neighbors:
                hq.heappush(bd, [seq[n], n[0], n[1], n[2]])
                edge[n[0], n[1], n[2]] = True
        step += 1
    return trapped


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt

    # %%
    im = ~ps.generators.random_spheres([400, 200], r=20, seed=0, clearance=10)

    inlets = np.zeros_like(im)
    inlets[0, :] = True
    inlets = inlets*im
    pc = capillary_transform(im)
    ip = invasion(im, pc=pc, inlets=inlets)

    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    outlets = outlets*im
    ps.tools.tic()
    trapped_new = find_trapped_regions(seq=ip.im_seq, outlets=outlets, return_mask=False)
    ps.tools.toc()
    ps.tools.tic()
    trapped = ps.filters.find_trapped_regions(seq=ip.im_seq, outlets=outlets, bins=None, return_mask=False)
    ps.tools.toc()


    # %%
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ip.im_seq/im, origin='lower', interpolation='none')
    ax[1].imshow(trapped/im, origin='lower', interpolation='none')
    ax[2].imshow(trapped_new/im, origin='lower', interpolation='none')


















