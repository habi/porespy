import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from ttictoc import tic, toc

ps.settings.tqdm['leave'] = True
np.random.seed(0)


t_orig = []
t_new = []
for s in [100]:
    im = ps.generators.blobs([s, s], porosity=0.7, blobiness=1)
    inlets = np.zeros_like(im)
    inlets[0, :] = True

    # Compare speeds
    tic()
    ibip_orig = ps.filters.ibip(im=im, inlets=inlets)
    t_orig.append(toc())
    tic()
    ibip_new = ps.simulations.invasion(im=im, inlets=inlets, voxel_size=1e-4)
    t_new.append(toc())

plt.plot(np.ones_like(t_orig)*np.array([200, 400, 600, 800, 1000])**2, t_orig, 'b.-')
plt.plot(np.ones_like(t_new)*np.array([200, 400, 600, 800, 1000])**2, t_new, 'r.-')
