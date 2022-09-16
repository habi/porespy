import numpy as np
from porespy.filters import local_thickness, find_trapped_regions
from porespy.filters import size_to_satn, size_to_seq, seq_to_satn
from porespy.filters import trim_disconnected_blobs
from porespy.tools import Results
from porespy.tools import get_tqdm


__all__ = [
    'imbibition',
]


tqdm = get_tqdm()


def imbibition(im, inlets=None, outlets=None, residual=None, lt=None,
                    sigma=0.072, theta=180, voxel_size=1):
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
    lt : ndarray, optional
        The local thickness of the void space.  If not provided it will be
        generated using the default values. Providing one if available saves
        time.

    Notes
    -----
    The simulate proceeds as though the non-wetting phase pressure is very
    high and is slowly lowered.  The imbibition occurs into the smallest
    accessible regions.

    Examples
    --------

    """

    if lt is None:
        lt = local_thickness(im=im)
    sizes = np.zeros_like(lt)
    for i, r in tqdm(enumerate(np.unique(lt)[1:])):
        imtemp = (lt <= r)*im
        if inlets is not None:
            if residual is not None:
                tmp = imtemp + residual
            else:
                tmp = np.copy(imtemp)
            tmp = trim_disconnected_blobs(tmp, inlets=inlets)
            imtemp = imtemp * tmp
        if residual is not None:
            imtemp += residual
        sizes += (sizes == 0)*(imtemp * r)

    seq = size_to_seq(size=-sizes, im=im)*(sizes > 0)
    if outlets is not None:
        trapped = find_trapped_regions(seq=seq, outlets=outlets)
        sizes[trapped] = 0
        seq[trapped] = 0
    else:
        trapped = None
    satn = (1 - seq_to_satn(seq))
    satn[sizes == 0] = np.inf
    satn = satn*im
    im_pc = 2*sigma*np.cos(np.deg2rad(theta))/(sizes*voxel_size)

    sz = np.unique(sizes)[1:]
    p = []
    s = []
    for n in sz:
        r = n*voxel_size
        pc = -2*sigma*np.cos(np.deg2rad(theta))/r
        p.append(pc)
        temp = 1 - ((sizes <= n)*(sizes > 0)).sum()/im.sum()
        s.append(temp)

    result = Results()
    result.__doc__ = 'This docstring should be customized to describe attributes'
    result.im = im
    result.im_sizes = sizes
    result.im_seq = seq
    result.im_snwp = satn
    result.im_trapped = trapped
    result.pc = p
    result.snwp = s

    return result


if __name__ == '__main__':
    import porespy as ps
    import matplotlib.pyplot as plt
    np.random.seed(0)
    vx = 1e-5
    im = ps.generators.blobs(shape=[800, 800], porosity=0.7, blobiness=1.5)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True

    # Perform imbibition with no trapping
    imb1 = imbibition(im=im, inlets=inlets, voxel_size=vx)
    fig, g = plt.subplots(1, 1)
    g.set_xlabel('log(Capillary Pressure) (P_nwp - P_wp) [Pa]')
    g.set_ylabel('Non-Wetting Phase (nwp) Saturation')
    g.step(np.log10(imb1.pc), imb1.snwp, 'b-o', where='post', label='imb wo trapping')
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(imb1.im_sizes/im, origin='lower')
    # ax[1].imshow(imb1.im_seq/im, origin='lower')
    # ax[2].imshow(imb1.im_snwp/im, origin='lower')

    # Perform imbibition WITH trapping
    imb2 = imbibition(im=im, inlets=inlets, outlets=outlets, voxel_size=vx)
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(imb2.im_sizes/im, origin='lower')
    # ax[1].imshow(imb2.im_seq/im, origin='lower')
    # ax[2].imshow(imb2.im_snwp/im, origin='lower')
    g.step(np.log10(imb2.pc), imb2.snwp, 'r-o', where='post', label='imb w trapping')

    # %% Perform imbibition with residual
    # First get residual from a drainage simulation
    drn1 = ps.simulations.drainage(im, inlets=inlets, voxel_size=vx)
    drn2 = ps.simulations.drainage(im, inlets=inlets, outlets=outlets, voxel_size=vx)
    # fig, ax = plt.subplots(1, 3)
    # ax[1].imshow(drn.im_pc/im, origin='lower')
    # ax[2].imshow(drn.im_satn/im, origin='lower')
    g.step(np.log10(drn1.pc), drn1.snwp, 'y-o', where='post', label='drn wo trapping')
    g.step(np.log10(drn2.pc), drn2.snwp, 'g-o', where='post', label='drn w trapping')

    # Now use trapped wetting phase in imbibition
    imb3 = imbibition(im=im, inlets=inlets, residual=drn2.im_trapped, voxel_size=vx)
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(imb3.im_sizes/im, origin='lower')
    # ax[1].imshow(imb3.im_seq/im, origin='lower')
    # ax[2].imshow(imb3.im_snwp/im, origin='lower')
    g.step(np.log10(imb3.pc), imb3.snwp, 'm--o', where='post', label='2nd imb wo trapping')

    imb4 = imbibition(im=im, inlets=inlets, outlets=outlets, residual=drn2.im_trapped, voxel_size=vx)
    g.step(np.log10(imb4.pc), imb4.snwp, 'c--o', where='post', label='2nd imb w trapping')
    g.legend()




























