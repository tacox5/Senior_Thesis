import numpy as np, os
from glob import glob
from scipy.interpolate import LinearNDInterpolator,interp1d
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.cosmology import WMAP9 as cosmo

f0 = 1420405751.7667  # Hz

def build_model_interp(parm_array,delta2_array,k_array,redshift,
    regrid_ks=None):
    #input an array of models, the parameters for each model, list of k modes
    #   and the desired redshift
    #return a list of ks and a list of matching interpolation functions,
    # function has call f(log10(Nx),alphaX,log10(MminX))
    #parm_array expected to be nmodels,nparms
    #with columns (z,Nf,Nx,alphaX,MminX,other-stuff....)
    #delta2_array expected to be nmodels,nkmodes
    #NOTE: assumes all models are computed at the same k modes
    closest_redshift = parm_array[np.abs(parm_array[:,0]-redshift).argmin(),0]
    model_points = parm_array[parm_array[:,0]==closest_redshift,2:5]

    #interpolate NX and Mmin in log space
    model_points[:,0] = np.log10(model_points[:,0])
    model_points[:,2] = np.log10(model_points[:,2])
    #get the power spectrum values that go with this redshift
    raw_model_values = delta2_array[parm_array[:,0]==closest_redshift]
    model_values = [[]]*len(raw_model_values)
    if not regrid_ks is None: #regrid to a different set of k bins
        for i in xrange(len(raw_model_values)):
            model_values[i] = interp1d(k_array,raw_model_values[i,:])(regrid_ks)
        model_values = np.array(model_values)
        k_array = regrid_ks
        print "interpolated sim shape after regridding",model_values.shape
    else:
        model_values = raw_model_values.copy()
    #for a single redshift, build an interplation for each k mode
    Pk_models_atz = []
    for ki,k in enumerate(k_array):

        M = LinearNDInterpolator(model_points,model_values[:,ki])
        Pk_models_atz.append(M)
    return Pk_models_atz
def build_tau_interp_model(parm_array):
    #interpolate NX and Mmin in log space
    alphaXs = np.sort(list(set(parm_array[:,3])))
    Mmins = np.sort(list(set(parm_array[:,4])))
    Nxs = np.sort(list(set(parm_array[:,2])))
    taus = []
    for Nx in Nxs:
        for alphaX in alphaXs:
            for Mmin in Mmins:
                _slice = np.argwhere(all_and([
                                    parm_array[:,2]==Nx,
                                    parm_array[:,3]==alphaX,
                                    parm_array[:,4]==Mmin]
                                    ))
                taus.append([np.log10(Nx),alphaX,np.log10(Mmin),
                            nf_to_tau(parm_array[_slice,0].squeeze(),
                            parm_array[_slice,1].squeeze())])
    taus = np.array(taus)
    return LinearNDInterpolator(taus[:,:3],taus[:,3])

def all_and(arrays):
    #input a list or arrays
    #output the arrays anded together
    if len(arrays)==1:return arrays
    out = arrays[0]
    for arr in arrays[1:]:
        out = np.logical_and(out,arr)
    return out

def load_andre_models(fileglob, sortz=True):
    #input a string that globs to the list of input model files
    #return arrays of parameters,k modes, delta2,and delt2 error
    #parm_array expected to be nmodels,nparms
    #with columns (z,Nf,Nx,alphaX,Mmin,other-stuff....)
    #delta2_array expected to be nmodels,nkmodes
    filenames = glob(fileglob)
    parm_array = []
    k_array = []
    delta2_array = []
    delta2_err_array = []
    for filename in filenames:
        parms = os.path.basename(filename).split('_')
        if parms[0].startswith('reion'):continue
        parm_array.append(map(float,[parms[3][1:],
                            parms[4][2:],  # Nf
                            parms[6][5:],  # zetaX
                            parms[7][-3:],  # alphaX
                            parms[8][8:],  # TvirminX
                            parms[9][5:]]))  # aveTb
        D = np.loadtxt(filename)
        k_array.append(D[:,0])
        delta2_array.append(D[:,1])
        delta2_err_array.append(D[:,2])
    parm_array = np.array(parm_array)
    raw_parm_array = parm_array.copy()
    k_array = np.ma.array(k_array)
    raw_k_array = k_array.copy()
    delta2_array = np.ma.masked_invalid(delta2_array)
    raw_delta2_array = delta2_array.copy()
    delta2_err_array = np.ma.array(delta2_err_array)
    if sortz:
        # Sort by redshift
        order = np.argsort(parm_array[:, 0])
        parm_array = parm_array[order, :]
        k_array = k_array[order, :]
        delta2_array = delta2_array[order, :]
        delta2_err_array = delta2_err_array[order, :]
    return parm_array,k_array,delta2_array,delta2_err_array

def load_andre_global_models(fileglob):
    #input a string that globs to the list of input model files
    #return a concatenated array of parameters
    #columns Nx,alphaX,Mmin
    # and global histories
    #dimensions len(parms) x n_redshifts x nparms
    #there are other columns that I don't know what they are
    filenames = glob(fileglob)
    parm_array = []
    global_evolution = []
    for filename in filenames:
        if not os.path.basename(filename).startswith('global'): continue
        parms = os.path.basename(filename).split('_')
        parm_array.append(map(float,
                            [parms[5][2:], #Nx
                             parms[6][-3:], #alphaX
                             parms[7][5:], #Mmin
                            ]))
        D = np.loadtxt(filename)
        global_evolution.append(D)
    return np.array(parm_array),global_evolution

def nf_to_tau(z,nf):
    #based on Liu et al 1509.08463
    """
    i) Take your ionization history, x_{HII} (z).
    ii) Numerically compute the integral \int_0^{zCMB} dz x_{HII} (1+z)^2 / sqrt{OmL + Omm (1+z)^3}.
    I would take OmL = 1 - Omm and Omm = 0.3089.
    In practice the integral doesn't need to be literally taken to zCMB since the ionization fraction is basically zero well before you hit those redshifts
    iii) Multiply by 0.00210228. This includes all the constants in Eq. (10) of my paper.
    iv) Add 0.001. That accounts for helium reionization (if you want to be more precise, it's 0.001223 for helium reionization happening at z = 3.5 and 0.000986 for z = 3.0).
    v) You have tau_CMB!
    """
    Omm = 0.3089
    Oml = 1 - Omm
    coeff = 0.00210228 #this includes all the constants out front of eq 10
    z = np.concatenate([np.linspace(0,z.min()),z])
    nf = np.concatenate([np.zeros(50),nf])
    xHI = interp1d(z,1-nf)
    E = lambda z: xHI(z) * (1+z)**2 / np.sqrt(Oml + Omm * (1+z)**3)
    tau_H  = integrate.quad(E,z.min(),z.max())[0]*coeff
    tau_He = 0.001223 #That accounts for helium reionization
    tau = tau_H + tau_He
    return tau


def compare_runs(runs, labels=None):
    # Input list of file globs to 21cmfast runs
    # Will generate some plots to compare the runs
    nruns = len(runs)
    if labels is None:
        labels = range(nruns)
    if len(labels) < nruns:
        labels += range(len(labels), nruns)
    parms, ks, delta2s, errs = [], [], [], []
    for run in runs:
        temp = load_andre_models(run)
        order = np.argsort(temp[0][:, 0])  # sort by redshift
        parms.append(temp[0][order, :])
        ks.append(temp[1][order, :])
        delta2s.append(temp[2][order, :])
        errs.append(temp[3][order, :])

    plt.figure('Comparison')
    plt.clf()
    handles = []
    lowk, highk = 0.1, 2.0
    midz = (parms[0][-1, 0] + parms[0][0, 0]) / 2.0
    for run in xrange(nruns):
        plt.subplot(221)
        freqs = f0 / (parms[run][:, 0] + 1) * 1e-6  # MHz
        handles += plt.plot(parms[run][:, 0], parms[run][:, 5], label=labels[run])
        plt.subplot(222)
        plt.plot(parms[run][:, 0], parms[run][:, 1], label=labels[run])
        plt.subplot(223)
        # Plot a PS at mid redshift
        ind = np.argmin(np.abs(parms[run][:, 0] - midz))
        plt.errorbar(ks[run][ind, :], delta2s[run][ind, :], yerr=errs[run][ind, :])
        plt.subplot(224)
        # Plot small and large scale power vs redshift
        ind = np.argmin(np.abs(ks[run][0, :] - lowk))
        temp, = plt.plot(parms[run][:, 0], delta2s[run][:, ind])
        ind = np.argmin(np.abs(ks[run][0, :] - highk))
        plt.plot(parms[run][:, 0], delta2s[run][:, ind], '--', color=temp.get_color())

    plt.subplot(221)
    plt.title('Tave vs z')
    plt.legend(handles=handles, loc=4)
    plt.gca().invert_xaxis()
    plt.subplot(222)
    plt.title('Neutral fraction vs z')
    plt.subplot(223)
    plt.title('PS at z = ' + str(midz))
    plt.loglog()
    plt.subplot(224)
    plt.title('Large and Small scale PS')
    plt.semilogy()


def view_global_xray_runs(dirglob):
    # Take fileglob pointing to a series of runs and make some plots to compare them
    # dirglob should point to the directories within Deldel_T_power_spec output from 21cmfast
    dirs = glob(dirglob)
    nruns = len(dirs)
    Nx = []
    Mmin = []
    runs = []
    data = []
    for dir in dirs:
        parms = os.path.basename(dir).split('_')
        Nx.append(float(parms[0][2:]))
        Mmin.append(float(parms[1][4:]))
        runs.append(dir + '/*')  # used to pass to load_andre_models
        p = load_andre_models(runs[-1])[0]
        order = np.argsort(p[:, 0])
        data.append(np.array([p[order, 0], p[order, 5]]))

    # Now sort Nx and Mmin
    Mmin = np.array(Mmin)
    Nx = np.array(Nx)
    Nxu = np.sort(np.array(list(set(Nx))))  # Get unique values, convert back to list, then to np array
    Mminu = np.sort(np.array(list(set(Mmin))))
    zs = []
    Tbs = []
    for i in xrange(len(Nxu)):
        ind = np.where(Nx == Nxu[i])[0]
        order = np.argsort(Mmin[ind])
        zs.append([])
        Tbs.append([])
        for j in ind[order]:
            zs[i].append(data[j][0, :])
            Tbs[i].append(data[j][1, :])

    return Nxu, Mminu, zs, Tbs


def get_global_reion_runs(dirglob):
    # Take fileglob pointing to a series of runs and return relevant global data
    # dirglob should point to the directories within Deldel_T_power_spec output from 21cmfast
    dirs = glob(dirglob)
    nruns = len(dirs)
    Zeta = []
    Tvir = []
    mfp = []
    runs = []
    data = []
    for dir in dirs:
        parms = os.path.basename(dir).split('_')
        Zeta.append(float(parms[0][4:]))
        Tvir.append(float(parms[1][4:]))
        mfp.append(float(parms[2][3:]))
        runs.append(dir + '/*')  # used to pass to load_andre_models
        p = load_andre_models(runs[-1])[0]
        order = np.argsort(p[:, 0])
        data.append(np.array([p[order, 0], p[order, 1], p[order, 5]]))

    # Now sort parameters
    Zeta = np.array(Zeta)
    Tvir = np.array(Tvir)
    mfp = np.array(mfp)
    Zetau = np.sort(np.array(list(set(Zeta))))  # Get unique values, convert back to list, then to np array
    Tviru = np.sort(np.array(list(set(Tvir))))
    mfpu = np.sort(np.array(list(set(mfp))))
    zs = []
    nf = []
    Tbs = []
    for i in xrange(len(Zetau)):
        zs.append([])
        nf.append([])
        Tbs.append([])
        for j in xrange(len(Tviru)):
            ind = np.where((Zeta == Zetau[i]) & (Tvir == Tviru[j]))[0]
            order = np.argsort(mfp[ind])
            zs[i].append([])
            nf[i].append([])
            Tbs[i].append([])
            for k in ind[order]:
                zs[i][j].append(data[k][0, :])
                nf[i][j].append(data[k][1, :])
                Tbs[i][j].append(data[k][2, :])

    zs = np.array([np.array([np.array([np.array(x) for x in y]) for y in z]) for z in zs])
    nf = np.array([np.array([np.array([np.array(x) for x in y]) for y in z]) for z in nf])
    Tbs = np.array([np.array([np.array([np.array(x) for x in y]) for y in z]) for z in Tbs])
    return Zetau, Tviru, mfpu, zs, nf, Tbs


def plot_global_reion_runs(zs, data):
    # Take data (e.g. from get_global_reion_runs) and plot.
    nzeta = zs.shape[0]
    ntvir = zs.shape[1]
    nmfp = zs.shape[2]
    plt.figure('Global runs')
    plt.clf()
    plt.subplot(131)
    colors = iter(cm.copper(np.linspace(0, 1, nzeta)))
    for i in xrange(nzeta):
        plt.plot(zs[i, ntvir / 2, nmfp / 2, :], data[i, ntvir / 2, nmfp / 2, :], color=next(colors))
    plt.subplot(132)
    colors = iter(cm.copper(np.linspace(0, 1, ntvir)))
    for i in xrange(ntvir):
        plt.plot(zs[nzeta / 2, i, nmfp / 2, :], data[nzeta / 2, i, nmfp / 2, :], color=next(colors))
    plt.subplot(133)
    colors = iter(cm.copper(np.linspace(0, 1, nmfp)))
    for i in xrange(nmfp):
        plt.plot(zs[nzeta / 2, ntvir / 2, i, :], data[nzeta / 2, ntvir / 2, i, :], color=next(colors))


def build_light_cone(fileglob, zs=np.array([6, 6.5, 7]), boxtype='delta_T'):
    zs = np.array(zs)
    files = glob(fileglob)
    zind = {'delta_T': 5, 'Ts_z': 1, 'xH': 2, 'deltax': 3}
    zsim = []
    dims = []
    lengths = []
    files_keep = []
    for f in files:
        if not os.path.basename(f).startswith(boxtype):
            if not 'updated_smoothed_' +  boxtype in os.path.basename(f):
                continue
        files_keep.append(f)
        parms = os.path.basename(f).split('_')
        zsim.append(np.float(parms[zind[boxtype]][1:]))  # redshifts
        dims.append(np.int(parms[-2]))  # dim of box
        lengths.append(np.float(parms[-1][0:-3]))  # length in Mpc
    files = files_keep
    if (np.max(np.diff(dims)) > 0) or (np.max(np.diff(lengths)) > 0):
        raise(ValueError('Boxes are not all the same size'))
    if (np.max(zs) > np.max(zsim)) or (np.min(zs) < np.min(zsim)):
        raise(ValueError('Requested redshifts outside range of sim.'))
    order = np.argsort(zsim)
    files = [files[i] for i in order]
    zsim = np.array([zsim[i] for i in order])
    zsim0 = zsim[0]
    dim = dims[0]
    length = lengths[0]
    dx = length / dim

    lightcube = np.zeros((dim, dim, len(zs)), dtype=np.float32)
    Ds = cosmo.comoving_distance(zs).value - cosmo.comoving_distance(zsim0).value
    pix1 = map(int, np.floor(Ds / dx) % dim)
    wp1 = Ds / dx - np.floor(Ds / dx)
    pix2 = map(int, np.ceil(Ds / dx) % dim)
    wp2 = 1 - wp1
    box1 = [np.argmax(zsim[zsim <= z]) for z in zs]
    box2 = [i + 1 for i in box1]
    wb2 = (zs - zsim[box1]) / (zsim[box2] - zsim[box1])
    wb1 = (zsim[box2] - zs) / (zsim[box2] - zsim[box1])

    for i, z in enumerate(zs):
        data = np.fromfile(files[box1[i]], dtype=np.float32)
        data = data.reshape((dim, dim, dim))
        slice11 = data[:, :, pix1[i]]
        slice12 = data[:, :, pix2[i]]
        data = np.fromfile(files[box2[i]], dtype=np.float32)
        data = data.reshape((dim, dim, dim))
        slice21 = data[:, :, pix1[i]]
        slice22 = data[:, :, pix2[i]]
        lightcube[:, :, i] = (slice11 * wb1[i] * wp1[i] + slice12 * wb1[i] * wp2[i] +
                              slice21 * wb2[i] * wp1[i] + slice22 * wb2[i] * wp2[i])
    return lightcube


def build_history_cube(fileglob, z0=6.0, zs=np.array([6, 6.5, 7]), boxtype='delta_T'):
    """
    Build a cube of local history for fixed perp-slice (as opposed to lightcone,
    which follows light path in z).
    """
    zs = np.array(zs)
    files = glob(fileglob)
    zind = {'delta_T': 5, 'Ts_z': 1, 'xH': 2}
    zsim = []
    dims = []
    lengths = []
    files_keep = []
    for f in files:
        if not os.path.basename(f).startswith(boxtype):
            continue
        files_keep.append(f)
        parms = os.path.basename(f).split('_')
        zsim.append(np.float(parms[zind[boxtype]][1:]))  # redshifts
        dims.append(np.int(parms[-2]))  # dim of box
        lengths.append(np.float(parms[-1][0:-3]))  # length in Mpc
    files = files_keep
    if (np.max(np.diff(dims)) > 0) or (np.max(np.diff(lengths)) > 0):
        raise(ValueError('Boxes are not all the same size'))
    if (np.max(zs) > np.max(zsim)) or (np.min(zs) < np.min(zsim)):
        raise(ValueError('Requested redshifts outside range of sim.'))
    order = np.argsort(zsim)
    files = [files[i] for i in order]
    zsim = np.array([zsim[i] for i in order])
    zsim0 = zsim[0]
    dim = dims[0]
    length = lengths[0]
    dx = length / dim

    cube = np.zeros((dim, dim, len(zs)), dtype=np.float32)
    dist = cosmo.comoving_distance(z0).value - cosmo.comoving_distance(zsim0).value
    pix1 = int(np.floor(dist / dx) % dim)
    wp1 = 1 - (dist / dx - np.floor(dist / dx))
    pix2 = pix1 + 1
    wp2 = 1 - wp1
    box1 = [np.argmax(zsim[zsim <= z]) for z in zs]
    box2 = [i + 1 for i in box1]
    wb2 = (zs - zsim[box1]) / (zsim[box2] - zsim[box1])
    wb1 = (zsim[box2] - zs) / (zsim[box2] - zsim[box1])

    for i, z in enumerate(zs):
        data = np.fromfile(files[box1[i]], dtype=np.float32)
        data = data.reshape((dim, dim, dim))
        slice11 = data[:, :, pix1]
        slice12 = data[:, :, pix2]
        data = np.fromfile(files[box2[i]], dtype=np.float32)
        data = data.reshape((dim, dim, dim))
        slice21 = data[:, :, pix1]
        slice22 = data[:, :, pix2]
        cube[:, :, i] = (slice11 * wb1[i] * wp1 + slice12 * wb1[i] * wp2 +
                         slice21 * wb2[i] * wp1 + slice22 * wb2[i] * wp2)
    return cube
