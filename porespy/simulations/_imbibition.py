import numpy as np
from porespy.filters import local_thickness, find_trapped_regions
from porespy.filters import size_to_satn, size_to_seq, seq_to_satn
from porespy.filters import trim_disconnected_blobs
from porespy.tools import Results


__all__ = [

]


def pc_curve_from_ibi(sizes, im, sigma, theta, voxel_size=1):
    r"""
    Generate capillary pressure curve data from an imbibition simulation

    Parameters
    ----------
    sizes : ndarray
        An image containing the meniscus radii as which each voxel was invaded
    im : ndarray
        A boolean image of the porous media wtih ``True`` indicating the
        void space
    sigma : scalar
        Surface tension of the fluid-fluid pair
    theta : scalar
        Contact angle
    voxel_size : scalar
        The resolution of the image, in units of m/voxel edge

    Returns
    -------
    results : dataclass
        A dataclass-like object with ``pc`` and ``snwp`` as attributes
    """
    sz = np.unique(sizes)
    sz = sz[sz > 0]
    x = []
    y = []
    for n in sz:
        r = n*voxel_size
        pc = -2*sigma*np.cos(np.deg2rad(theta))/r
        x.append(pc)
        snwp = 1 - ((sizes <= n)*(sizes > 0)).sum()/im.sum()
        y.append(snwp)
    result = Results()
    result.pc = x
    result.snwp = y
    return result


def imbibition(im, inlets=None, outlets=None, residual=None,
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

    lt = local_thickness(im)

    if lt is None:
        lt = local_thickness(im=im)
    sizes = np.zeros_like(lt)
    for i, r in enumerate(np.unique(lt)[1:]):
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

    # sizes = ibi(im=im, inlets=inlets, residual=residual, lt=lt)
    seq = size_to_seq(size=-sizes, im=im)
    if outlets is not None:
        trapped = find_trapped_regions(seq=seq, outlets=outlets)
        sizes[trapped] = -1
    else:
        trapped = None
    satn = size_to_satn(sizes)
    d = pc_curve_from_ibi(sizes=sizes, im=im, sigma=sigma,
                          theta=theta, voxel_size=voxel_size)

    result = Results()
    result.__doc__ = r"""
        This is a dataclass meant to hold the various values computed by this
        function as attributes.  The following documentation explains each
        attribute.

        Attributes
        ----------
        sizes : ndarray
            An image the same shape ``im`` with the value in each voxel
            indicating the sphere radii (in voxels) at which is was invaded by
            the wetting phase. This array is returned by the
            ``porespy.filters.ibi`` function. An image of the wetting fluid
            configuration at a given meniscus radii R can be obtained with
            a simple threshold: ``nwp = result.sizes < R``.
        satn : ndarray
            An image the same shape ``im`` with the value in each voxel
            indicating the saturation of the domain at the point which it was
            invaded by the wetting fluid. This array is obtained from the
            ``porespy.filters.size_to_satn`` function. An image of the
            non-wetting fluid configuration at a saturation S can be obtained
            with a simple threshold: ``nwp = result.satn < S``
        trapped : ndarray
            A boolean mask with ``True`` values indicating the locations of
            trapped defending phase.  If ``outlets`` were not given then this
            attribute will be ``None``.
        pc : ndarray
            The list of capillary pressures applied during drainage.
        snwp : ndarray
            The list of saturations obtained at each of the capillary
            pressures listed in ``result.pc``.

        """
    result.im = im
    result.sizes = sizes
    result.satn = satn
    result.trapped = trapped
    result.pc = d.pc
    result.snwp = d.snwp

    return result


if __name__ == '__main__':
    import porespy as ps
    import matplotlib.pyplot as plt
    im = ps.generators.blobs([300, 300])
    imb = imbibition(im)
    plt.imshow(imb.satn)
    pc = pc_curve_from_ibi(imb.sizes, im, sigma=0.072, theta=160)
    plt.plot(pc.pc, pc.snwp)
    drn = ps.simulations.drainage(im=im, voxel_size=1)
    plt.plot(drn.pc, drn.snwp, 'r-o')

































