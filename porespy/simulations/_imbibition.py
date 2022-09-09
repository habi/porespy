from porespy.filters import ibi, local_thickness, find_trapped_regions
from porespy.filters import size_to_satn, size_to_seq, seq_to_satn
from porespy.metrics import pc_curve_from_ibi
from porespy.tools import Results


def ibsi_imbibition(im, inlets=None, outlets=None, residual=None,
                    sigma=0.072, theta=180, voxel_size=1):
    r"""
    Performs a imbibition simulation using image-based sphere insertion

    This is a helper function that performs all the steps needed to generate
    a imbibition curve from the image. It consists of calls to
    ``local_thickness``, ``sizes_to_satn``, and ``pc_curve_from_mio``.
    Optionally it also calls ``size_to_seq`` and ``find_trapped_regions``
    to apply trapping if ``outlets`` is given.

    Parameters
    ----------
    %(filters.local_thickness)s
    %(metrics.pc_curve_from_ibi)s

    Examples
    --------
    >>> import porespy as ps
    >>> import numpy as np
    >>> im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=2)
    >>> inlets = np.zeros_like(im)
    >>> inlets[0, :] = True
    >>> outlets = np.zeros_like(im)
    >>> outlets[-1, :] = True
    >>> r = ps.dns.ibsi_imbibition(im=im, inlets=inlets, outlets=outlets)
    """

    lt = local_thickness(im)
    sizes = ibi(im=im, inlets=inlets, residual=residual, lt=lt)
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
