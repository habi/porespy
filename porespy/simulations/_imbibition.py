import numpy as np
from porespy.filters import local_thickness, find_trapped_regions
from porespy.filters import size_to_satn, size_to_seq, seq_to_satn
from porespy.filters import trim_disconnected_blobs
from porespy.metrics import pc_curve_from_ibi
from porespy.tools import Results


__all__ = [
    'ibi',
]


def ibi(im, inlets=None, residual=None, lt=None):
    r"""
    Simulate imbibition of a wetting phase into an image

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
        time, and allows for over the number of sizes.

    Returns
    -------
    imbibed : ndarray
        A numpy ndarray the same shape as ``im`` with voxel values indicating
        the radius at which it was reached by the imbibing fluid

    Notes
    -----
    The simulate proceeds as though the non-wetting phase pressure is very
    high and is slowly lowered.  The imbibition occurs into the smallest
    accessible regions.

    See Also
    --------
    size_to_satn
    pc_curve_from_ibi
    ibsi_imbibition

    """
    if lt is None:
        lt = local_thickness(im=im)
    imb = np.zeros_like(lt)
    sizes = np.unique(lt)[1:]
    for i, r in enumerate(sizes):
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
        imb += (imb == 0)*(imtemp * r)



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
    seq = size_to_seq(size=sizes, im=im, ascending=True)
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
    im = ps.generators.blobs([300, 300])
    imb = imbibition(im)


































