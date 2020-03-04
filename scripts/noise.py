import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
from astropy.cosmology import Planck15 as cosmo
from twentyonecmFAST import load_binary_data
from powerbox import get_power
from astropy import constants as const
import xcorr
import sys
import pickle
import copy
import os
import glob
import matplotlib as mpl
from py21cmsense import sensitivity as sense

c = 3e8 * u.m / u.second
"""

Parallel Mode Resolution

"""

def k_par_res(z, R_res = 41.):
    """k parallel mode resolution for SPHEREx
    """
    return 2 * np.pi * (R_res * cosmo.H(z) / (c * (1 + z))).to(u.Mpc ** -1)


"""

Perpendicular Mode Resolution

"""

def k_perp_res(z, x_pix = 6.2 * u.arcsecond):
    """Perpendicular Resolution for
    """
    theta = x_pix.to(u.radian)
    return 2 * np.pi / (cosmo.comoving_distance(z) * theta)


"""

Window Function to Introduce Resolution Error to SPHEREx

"""

def W(kperp, kpar, z):
    """Window function to handle resolution limitations of SPHEREx

    Parameters
    ----------
    kperp: float
    kpar: float
    z: float

    Returns:


    """
    kpar_res = k_par_res(z).value
    kperp_res = k_perp_res(z).value
    return np.exp((kperp / kperp_res) ** 2 + (kpar / kpar_res) ** 2)

def lyman_noise(ps_interp, kperp, kpar, z, thermal = True, sample = True,
                thermal_noise = 3e-20):
    """
    Noise contribution from SPHEREx-like experiment
    """
    k = np.sqrt(kperp ** 2 + kpar ** 2)
    var = 0
    nu = 2.47e15 / (1. + z)
    if sample:
        try:
            var += ps_interp(k)
        except ValueError:
            return np.inf

    if thermal:
        var += k ** 3 / (2 * np.pi ** 2) * V_pix(z).value * (nu * thermal_noise) ** 2 * W(kperp, kpar, z)

    return var

def x_var(kperp, kpar, pspec = xps_interp):
    """
    """
    k = np.sqrt(kperp ** 2 + kpar ** 2)
    try:
        return pspec(k)
    except ValueError:
        return np.inf
    return

def x_power_spec(ks):
    """
    """
    ps = []
    for k in ks:
        try:
            ps.append(xps_interp(k))
        except ValueError:
            ps.append(np.inf)
    return np.array(ps)

def V_pix(z, x_pix = 6.2 * u.arcsecond):
    """
    """
    return ((cosmo.kpc_comoving_per_arcmin(z) * x_pix) ** 2 / k_par_res(z)).to(u.Mpc ** 3) / 2.
