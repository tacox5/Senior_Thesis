import numpy as np

def calc_FT_cube(incube, x, y, z, invert=False):
    """
    Function to FT cube and calculate k axes.

    Args:
        incube: 3D input cube in image space.
        x: 1D array of x-coordinates. Assumed to be in Mpc and evenly spaced.
        y: 1D array of y-coordinates. Assumed to be in Mpc and evenly spaced.
        z: 1D array of z-coordinates. Assumed to be in Mpc and evenly spaced.
        invert: Invert FT (go back to image space). Default False.

    Returns:
        FT_cube: 3D cube, the fourier transform of incube. No jacobian is applied.
        kx: 1D array of kx coordinates. Units Mpc^-1.
        ky: 1D array of ky coordinates. Units Mpc^-1.
        kz: 1D array of kz coordinates. Units Mpc^-1.
    """

    if invert:
        FT_cube = np.fft.ifftn(np.fft.ifftshift(incube))
    else:
        FT_cube = np.fft.fftshift(np.fft.fftn(incube))

    # Get k-axes
    dkx = 2 * np.pi / (x.max() - x.min())
    dky = 2 * np.pi / (y.max() - y.min())
    dkz = 2 * np.pi / (z.max() - z.min())
    kx = dkx * (np.arange(len(x)) - len(x) / 2)
    ky = dky * (np.arange(len(y)) - len(y) / 2)
    kz = dkz * (np.arange(len(z)) - len(z) / 2)

    return FT_cube, kx, ky, kz

def calc_PS_3d(incube, x, y, z):
    """
    Function to calculate 3D power spectrum from an input cube

    Args:
        incube: 3D input cube in image space.
        x: 1D array of x-coordinates. Assumed to be in Mpc and evenly spaced.
        y: 1D array of y-coordinates. Assumed to be in Mpc and evenly spaced.
        z: 1D array of z-coordinates. Assumed to be in Mpc and evenly spaced.

    Returns:
        PS: 3D power spectrum. If inputcube is in mK, PS is in mK^2 Mpc^3
        kx: 1D array of kx coordinates. Units Mpc^-1.
        ky: 1D array of ky coordinates. Units Mpc^-1.
        kz: 1D array of kz coordinates. Units Mpc^-1.
    """
    # Get 3D PS
    PS, kx, ky, kz = calc_FT_cube(incube, x, y, z)
    jacobian = np.mean(np.diff(x)) * np.mean(np.diff(y)) * np.mean(np.diff(z))
    PS = np.abs(jacobian * PS)**2. / (x.max() - x.min()) / (y.max() - y.min()) / (z.max() - z.min())


    return PS, kx, ky, kz

def calc_PS_1d(incube, x, y, z, k_bin=1):
    """
    Function to calculate 1D power spectrum from an input cube

    Args:
        incube: 3D input cube in image space.
        x: 1D array of x-coordinates. Assumed to be in Mpc and evenly spaced.
        y: 1D array of y-coordinates. Assumed to be in Mpc and evenly spaced.
        z: 1D array of z-coordinates. Assumed to be in Mpc and evenly spaced.
        k_bin: Factor by which to bin up k. Default 1.

    Returns:
        PS: 1D power spectrum. If inputcube is in mK, PS is in mK^2 Mpc^3
        k: 1D array of k coordinates. Units Mpc^-1.
    """
    # Get 3D PS
    PS_3d, kx, ky, kz = calc_PS_3d(incube, x, y, z)

    # Get k matrix
    kxmat, kymat, kzmat = np.meshgrid(kx, ky, kz, indexing='ij')
    kmat = np.sqrt(kxmat**2 + kymat**2 + kzmat**2)

    # Form output axis
    dk = np.mean([np.mean(np.diff(kx)), np.mean(np.diff(ky)), np.mean(np.diff(kz))]) * k_bin
    k = np.arange(0, kmat.max(), dk)
    k_inds = np.digitize(kmat, k - 0.5 * dk)

    # Bin the PS
    PS = np.zeros(len(k))
    for i in range(len(k)):
        ind = np.where(k_inds == i)
        if len(ind[0]) == 0:
            continue
        PS[i - 1] = np.mean(PS_3d[ind])

    return PS, k

def calc_PS_2d(incube, x, y, z, kperp_bin=1, kpar_bin=1):
    """
    Function to calculate 2D power spectrum from an input cube

    Args:
        incube: 3D input cube in image space.
        x: 1D array of x-coordinates. Assumed to be in Mpc and evenly spaced.
        y: 1D array of y-coordinates. Assumed to be in Mpc and evenly spaced.
        z: 1D array of z-coordinates. Assumed to be in Mpc and evenly spaced.
        kperp_bin: Factor by which to bin up kperp. Default 1.
        kpar_bin: Factor by which to bin up kpar. Default 1.

    Returns:
        PS: 2D power spectrum. If inputcube is in mK, PS is in mK^2 Mpc^3
        kperp: 1D array of kperp coordinates. Units Mpc^-1.
        kpar: 1D array of kpar coordinates. Units Mpc^-1.
    """
    # Get 3D PS
    PS_3d, kx, ky, kz = calc_PS_3d(incube, x, y, z)

    # Get kperp matrix
    kxmat, kymat = np.meshgrid(kx, ky, indexing='ij')
    kperpmat = np.sqrt(kxmat**2 + kymat**2)

    # Form output axes
    dkperp = np.mean([np.mean(np.diff(kx)), np.mean(np.diff(ky))]) * kperp_bin
    kperp = np.arange(0, kperpmat.max(), dkperp)
    dkpar = np.mean(np.diff(kz)) * kpar_bin
    kz = np.abs(kz)  # Fold over
    kpar = np.arange(0, kz.max(), dkpar)
    kperpinds = np.digitize(kperpmat, kperp - 0.5 * dkperp)
    kparinds = np.digitize(kz, kpar - 0.5 * dkpar)
    kpar_ind_lookup = []
    for j in range(len(kpar)):
        ind_par = np.where(kparinds == j)[0]
        kpar_ind_lookup.append(ind_par)

    # Bin the PS
    PS = np.zeros((len(kperp), len(kpar)))
    for i in range(len(kperp)):
        ind_perp = np.where(kperpinds == i)
        if len(ind_perp[0]) == 0:
            continue
        for j in range(len(kpar)):
            ind_par = kpar_ind_lookup[j]
            if len(ind_par) == 0:
                continue
            PS[i - 1, j - 1] = np.mean(PS_3d[ind_perp][0, ind_par])

    return PS, kperp, kpar
