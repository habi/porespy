import logging
import numpy as np
import openpnm as op
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster
from edt import edt
from porespy.tools import Results, get_tqdm, subdivide
from porespy.filters import apply_chords, region_size, fill_blind_pores
from porespy import settings


logger = logging.getLogger(__name__)


__all__ = [
    'tortuosity_bt',
    'rev_tortuosity',
    'analyze_blocks',
]


tqdm = get_tqdm()
settings.loglevel = 50


def get_block_sizes(shape, block_size_range=[10, 100]):
    r"""
    Finds all viable block sizes between lower and upper limits

    Parameters
    ----------
    shape : sequence of int
        The [x, y, z] size of the image
    block_size_range : sequence of 2 ints
        The [lower, upper] range of the desired block sizes. Default is [10, 100]

    Returns
    -------
    sizes : ndarray
        All the viable block sizes in the specified range
    """
    Lmin, Lmax = block_size_range
    a = np.ceil(min(shape)/Lmax).astype(int)
    block_sizes = min(shape) // np.arange(a, 9999)  # Generate WAY more than needed
    block_sizes = np.unique(block_sizes[block_sizes >= Lmin])
    return block_sizes


def rev_tortuosity(im, block_sizes=None, **kwargs):
    r"""
    Computes data for a representative element volume plot based on tortuosity

    This function is a wrapper for `analyze_block`. It basically just determines
    the block sizes that should be analyzed, then collects the result into a
    single DataFrame.

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with `True` values indicating the phase
        of interest.
    block_sizes : sequence of ints
        The sizes of block to test.

    Returns
    -------
    df : DataFrame
        A `pandas` DataFrame with the tortuosity and volume for each block, along
        with other useful data like the porosity.

    Notes
    -----
    DataFrames can be concatentated using `pandas.concat((df1, df2))`, so the
    results can be combined easily if calling this function multiple times.
    """
    block_sizes = np.array(block_sizes, dtype=int)
    tau = []
    disable = settings.tqdm['disable']
    msg = "Scanning block sizes"
    for s in tqdm(block_sizes, desc=msg, **settings.tqdm):
        settings.tqdm['disable'] = True
        tau.append(analyze_blocks(im, block_size=s, **kwargs))
    settings.tqdm['disable'] = disable
    df = pd.concat(tau)
    client = dask.distributed.client._get_global_client()
    if client is not None:
        client.cluster.close()
        client.close()
    return df


def calc_g(im, axis, solver_args={}):
    r"""
    Calculates diffusive conductance of an image in the direction specified

    Parameters
    ----------
    image : ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    solver_args : dict
        Dicionary of keyword arguments to pass on to the solver.  The most
        relevant one being `'tol'` which is 1e-6 by default. Using larger values
        might improve speed at the cost of accuracy.

    Returns
    -------
    results : dataclass-like
        An object with the results of the calculation as attributes.
    """
    from porespy.simulations import tortuosity_fd
    solver_args = {'tol': 1e-6} | solver_args
    solver = solver_args.pop('solver', None)
    try:
        solver = op.solvers.PyamgRugeStubenSolver(**solver_args)
        results = tortuosity_fd(im=im, axis=axis, solver=solver)
    except Exception:
        results = Results()
        results.effective_porosity = 0.0
        results.original_porosity = im.sum()/im.size
        results.tortuosity = np.inf
    L = im.shape[axis]
    A = np.prod(im.shape)/im.shape[axis]
    g = (results.effective_porosity * A) / (results.tortuosity * (L - 1))
    results.diffusive_conductance = g
    results.volume = np.prod(im.shape)
    return results


def estimate_block_size(im, scale_factor=3, mode='radial'):
    r"""
    Predicts suitable block size based on properties of the image

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    scale_factor : int
        The factor by which to increase the estimated block size to ensure blocks
        are big enough
    mode : str
        Which method to use when estimating the block size. Options are:

        ======== ====================================================================
        mode     description
        ======== ====================================================================
        radial   Uses the maximum of the distance transform. This means that the
                 value will be unable to see anisotropy and will only detect the
                 smallest distance.
        linear   Draws chords drawn in all three directions, then uses the median
                 length. If the same is anisotropic this will be a better choice
                 since it will see that longer distances of the elongated direction.
        ======== ====================================================================

    Returns
    -------
    block_size : list of ints
        The block size in a 3 directions. If `mode='radial'` these will all be equal.
        If `mode='linear'` they could differ in each direction.  The largest block
        size will be used if passed into another function.
    """
    if mode.startswith('rad'):
        dt = edt(im, parallel=settings.ncores)
        x = min(dt.max()*scale_factor, min(im.shape)/2)  # Div by 2 is fewest blocks
        size = np.array([x, x, x], dtype=int)
    elif mode.startswith('lin'):
        size = []
        for ax in range(im.ndim):
            crds = apply_chords(im, axis=ax)
            crd_sizes = region_size(crds > 0)
            L = np.median(crd_sizes[crds > 0])/2  # Divide by 2 to make it a radius
            L = min(L*scale_factor, im.shape[ax]/2)  # Div by 2 is fewest blocks
            size.append(L)
        size = np.array(size, dtype=int)
    else:
        raise Exception('Unsupported mode')
    return size


def block_size_to_divs(shape, block_size):
    r"""
    Finds the number of blocks in each direction given the size of the blocks

    Parameters
    ----------
    shape : sequence of ints
        The [x, y, z] shape of the image
    block_size : int or sequence of ints
        The size of the blocks

    Returns
    -------
    divs : list of ints
        The number of blocks to divide the image into along each axis. The minimum
        number of blocks is 2.
    """
    shape = np.array(shape)
    divs = shape // np.array(block_size)
    # scraps = shape % np.array(block_size)
    divs = np.clip(divs, a_min=2, a_max=shape)
    return divs


def analyze_blocks(im, block_size, dask_args={}, solver_args={}):
    r"""
    Computes the diffusive conductance tensor for each block and returns values in a
    DataFrame

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or sequence of ints
        The size of the blocks to use. Only cubic blocks are supported so an integer
        must be given, or an exception is raised. If the image is not evenly
        divisible by the given `block_size` any extra voxels are removed from the
        end of each axis before all processing occcurs.
    dask_args : dict
        A dictionary of arguments to control the behavior of `dask`. Options are:

        =========== =================================================================
        key         values
        =========== =================================================================
        'enable'    Can be `True` (default) or `False`. If `True` then `dask` is
                    used (obviously), but can be set to `False` for timing tests,
                    unit tests, or if you're willing to wait.
        'client'    If `False` (default) then first a check is performed to see
                    if a client is already running, if not then a `LocalCluster`
                    is started up.
        'close'     If `True` (default) then the client is closed upon completion.
                    Keeping it open would be preferred if many calls to this
                    function are expected, like doing REV calculations.
        =========== =================================================================

    solver_args : dict
        A dictionary of arguments to control the solver used. This dictionary is
        passed to `porespy.simulations.tortuosity_fd` function.

    Returns
    -------
    df : DataFrame
        A `pandas` data frame with the properties for each block on a given row.

    """
    # Deal with dask args and clients
    dask_args = {'enable': True, 'client': None, 'close': False} | dask_args
    solver_args = {'tol': 1e-6} | solver_args
    if dask_args['enable'] is True:
        if dask_args['client'] is None:  # If client is provided, use it
            client = dask.distributed.client._get_global_client()  # Use it if found
            if client is None:  # Otherwise create one
                client = Client(LocalCluster(silence_logs=logging.CRITICAL))

    # Parse args
    if not np.isscalar(block_size):
        raise Exception('Only cubic blocks supported. block_size must be a scalar')

    # Trim image to be a clean multiple of block_size in all directions
    im_temp = np.copy(im)
    if np.any(np.array(im_temp.shape) % block_size):
        tmp1 = im_temp.shape
        stops = block_size * (np.array(im_temp.shape) // block_size)
        s = tuple(slice(0, stops[i], None) for i in range(im_temp.ndim))
        im_temp = im_temp[s]
        tmp2 = im_temp.shape
        msg = f"Image was trimmed to fit given block size from {tmp1} to {tmp2}"
        logger.warn(msg)

    # Prepare to find tau for all blocks in all principal directions
    df = pd.DataFrame()
    offset = int(block_size/2)
    msg = "Analyzing blocks along each axis"
    for ax in tqdm(range(im.ndim), desc=msg, **settings.tqdm):
        # Rotate, remove 1/2 block from start and end, and rotate back
        tmp = np.swapaxes(im_temp, 0, ax)
        tmp = tmp[offset:-offset, ...]
        tmp = np.swapaxes(tmp, 0, ax)
        slices = subdivide(tmp, block_size=block_size, mode='offset')
        if dask_args['enable']:  # Do it with dask
            queue = []
            for s in slices:
                queue.append(dask.delayed(calc_g)(im_temp[s],
                                                  axis=ax,
                                                  solver_args=solver_args))
            results = dask.compute(*queue, scheduler=client)
        else:  # Or do it the simple way
            results = []
            for s in slices:
                results.append(calc_g(im_temp[s], axis=ax, solver_args=solver_args))

        # Convert results to a new DataFrame and append to main DataFrame
        df_temp = pd.DataFrame()
        df_temp['eps_orig'] = [r.original_porosity for r in results]
        df_temp['eps_perc'] = [r.effective_porosity for r in results]
        df_temp['g'] = [r.diffusive_conductance for r in results]
        df_temp['tau'] = [r.tortuosity for r in results]
        df_temp['volume'] = [r.volume for r in results]
        df_temp['length'] = [block_size for r in results]
        df_temp['axis'] = [ax for _ in results]
        df_temp['time'] = [r.time for r in results]
        df = pd.concat((df, df_temp))

    if dask_args['enable'] and dask_args['close']:
        client.cluster.close()
        client.close()
    return df


def network_to_tau(df, im, block_size):
    r"""
    Compute the tortuosity of a network populated with diffusive conductance values
    from the given dataframe.

    Parameters
    ----------
    df : dataframe
        The dataframe returned by the `blocks_to_dataframe` function
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int
        The size of the blocks used to compute the conductance values in `df`

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """
    if not np.isscalar(block_size):
        raise Exception('Only cubic blocks supported. block_size must be a scalar')
    divs = block_size_to_divs(shape=im.shape, block_size=block_size)
    net = op.network.Cubic(shape=divs)
    phase = op.phase.Phase(network=net)
    gx = np.array(df['g'])[df['axis'] == 0]
    gy = np.array(df['g'])[df['axis'] == 1]
    gz = np.array(df['g'])[df['axis'] == 2]
    g = np.hstack((gz, gy, gx))  # throat indices in openpnm are in reverse order!
    g[g == 0] = 1e-20
    phase['throat.diffusive_conductance'] = g
    bcs = {0: {'in': 'left', 'out': 'right'},
           1: {'in': 'front', 'out': 'back'},
           2: {'in': 'top', 'out': 'bottom'}}
    im_temp = fill_blind_pores(im, surface=True)
    e = np.sum(im_temp, dtype=np.int64) / im_temp.size
    D_AB = 1
    tau = []
    for ax in range(im.ndim):
        fick = op.algorithms.FickianDiffusion(network=net, phase=phase)
        fick.set_value_BC(pores=net.pores(bcs[ax]['in']), values=1.0)
        fick.set_value_BC(pores=net.pores(bcs[ax]['out']), values=0.0)
        fick.run()
        rate_inlet = fick.rate(pores=net.pores(bcs[ax]['in']))[0]
        L = im.shape[ax] - block_size
        A = np.prod(im.shape) / im.shape[ax]
        D_eff = rate_inlet * L / (A * (1 - 0))
        tau.append(e * D_AB / D_eff)
    ws = op.Workspace()
    ws.clear()
    return tau


def tortuosity_bt(im, block_size=None):
    r"""
    Computes the tortuosity tensor of an image using the "block and tackle"
    method

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int
        The size of the blocks to use. If not given then it will be estimated
        automatically.

    Returns
    -------
    tau : list of floats
        The tortuosity in all principal directions
    """
    if block_size is None:
        block_size = estimate_block_size(im, scale_factor=3, mode='radial')
    df = analyze_blocks(im, block_size, dask_args={'close': True})
    tau = network_to_tau(df=df, im=im, block_size=block_size)
    return tau


if __name__ =="__main__":
    import porespy as ps
    # im = ps.generators.cylinders(shape=[300, 200, 100], porosity=0.5, r=3, seed=1)
    im = ps.generators.blobs(shape=[256, 256, 256], porosity=0.99, blobiness=1, seed=1)
    ps.tools.tic()
    s = np.arange(16, 129, 16)
    df = rev_tortuosity(im, block_sizes=s, dask_args={'enable': False})
    t = ps.tools.toc()
    print(df)
    # block_size = estimate_block_size(im, scale_factor=3, mode='linear')
    # tau = tortuosity_bt(im=im, block_size=block_size)
    # print(tau)
