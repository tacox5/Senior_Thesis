import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

files = glob.glob('data/ccc_*npy')
files.sort()

k = np.load('data/ks.npy')

k_spaced = np.logspace(np.log10(k.min() + 0.005),  np.log10(k.max() - 0.005), 100)

zs = [6.0, 7.0, 8.0]

colors = plt.cm.BuPu(np.linspace(0.5, 1., 3))[::-1]
colors = ['#2F2E5F', '#427AB7', '#8CCBD6'][::-1]
ls = ['--', ':', '-.']

plt.figure(figsize=(10,6))

for i in np.arange(len(files) - 1):
    r = np.load(files[i])
    ps = interp1d(k, r, kind = 'linear')
    plt.plot(k_spaced, gaussian_filter(ps(k_spaced), 2.), color = colors[i], linestyle = ls[i],
             label = '$z =$ {}'.format(zs[i]), linewidth = 2)

plt.xscale('log')
plt.xlim([4e-2, 6])
plt.ylim([-1, 1])
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$k \left[h \ {\rm Mpc}^{-1} \right]$', fontsize = 18)
plt.ylabel('$r(k)$', fontsize = 18)
plt.axhline(0, color = 'k', linestyle = ':', zorder = 0)
plt.legend(fontsize = 14)
plt.show()
