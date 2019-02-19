#load the simulations
import sys,os
from glob import glob
from pylab import *
import numpy as n
import time
from matplotlib import colorbar as Colorbar
from matplotlib import rcParams,cm
import ipdb as PDB
rcParams['font.size'] = 14
def all_and(arrays):
    #input a list or arrays
    #output the arrays anded together
    if len(arrays)==1:return arrays
    out = arrays[0]
    for arr in arrays[1:]:
        out = n.logical_and(out,arr)
    return out
from twentyonecmfast_tools import build_model_interp,load_andre_models,build_tau_interp_model


    
#def load_andre_models(fileglob):
#    #input a string that globs to the list of input model files
#    #return arrays of parameters,k modes, delta2,and delt2 error
#    #parm_array expected to be nmodels,nparms
#    #with columns (z,Nf,Nx,alphaX,Mmin,other-stuff....)
#    #delta2_array expected to be nmodels,nkmodes
#    filenames = glob(sys.argv[1])
#    parm_array = []
#    k_array = []
#    delta2_array = []
#    delta2_err_array = []
#    for filename in filenames:
#        parms = os.path.basename(filename).split('_')
#        if parms[0].startswith('reion'):continue
#        parm_array.append(map(float,[parms[3][1:], #redshift
#                            parms[4][2:], #Nf (Neutral Fraction)
#                            parms[6][2:], #Nx
#                            parms[7][-3:], #alphaX
#                            parms[8][5:], #Mmin
#                            parms[9][5:]]))
#        D = n.loadtxt(filename)
#        k_array.append(D[:,0])
#        delta2_array.append(D[:,1])
#        delta2_err_array.append(D[:,2])
#    parm_array = n.array(parm_array)
#    raw_parm_array = parm_array.copy()
#    k_array = n.ma.array(k_array)
#    raw_k_array = k_array.copy()
#    delta2_array = n.ma.masked_invalid(delta2_array)
#    raw_delta2_array = delta2_array.copy()
#    delta2_err_array = n.ma.array(delta2_err_array)
#    return parm_array,k_array,delta2_array,delta2_err_array


#load the models and get the lists of parameters for the runs
parm_array,k_array,delta2_array,delta2_err_array = load_andre_models(sys.argv[1])

alphaXs = n.sort(list(set(parm_array[:,3])))
Mmins = n.sort(list(set(parm_array[:,4])))
Mmin_grid = parm_array[:,4]
Nxs = n.sort(list(set(parm_array[:,2])))
Nx_grid = parm_array[:,2]
myz = 10.9
#round to the nearest redshift bin
myz = parm_array[n.abs(parm_array[:,0]-myz).argmin(),0]

#build the interpolation model
T = time.time()
Pk_models_atz = build_model_interp(parm_array,delta2_array,k_array[0],myz)
print "interplation took ",time.time()-T,"s"



#examine the interpolation between two points
MminX_model_range = [3.1e9,3.1e9]
alphaX_model_range = [1.,1.]
NX_model_range = [1.0e-2,5.0e-2]
slices = []
figure()
for i in xrange(len(NX_model_range)):
    #select off the modelled spectrum for this parameter set
    _slice = n.argwhere(all_and([n.abs(parm_array[:,0]-myz)<0.05,
                            parm_array[:,2]==NX_model_range[i],
                            parm_array[:,3]==alphaX_model_range[i],
                            parm_array[:,4]==MminX_model_range[i]])).squeeze()
    print "_slice index",_slice
    slices.append(_slice)
    plot(k_array[_slice],delta2_array[_slice],'k.') #plot the two valid ends of the interp
MminX_models = n.linspace(MminX_model_range[0],MminX_model_range[1])
alphaX_models = n.linspace(alphaX_model_range[0],alphaX_model_range[1])
NX_models = n.logspace(n.log10(NX_model_range[0]),n.log10(NX_model_range[1]))
interpolated_parameter_space = n.vstack([NX_models,alphaX_models,MminX_models]).T
print interpolated_parameter_space[0]
print interpolated_parameter_space[-1]
for i in xrange(len(interpolated_parameter_space)):
    Pk = []
    _Nx,_alphaX,_MminX = interpolated_parameter_space[i]
    for ki in xrange(len(k_array[0])):
        Pk.append(Pk_models_atz[ki](n.log10(_Nx),_alphaX,n.log10(_MminX)))
    loglog(k_array[0],Pk,'b')
xlabel('k')
ylabel('$\\Delta^2$')
savefig('xray_interp_demo.png')

fig =figure(figsize=(11,4))
logNxs = n.linspace(-2,1.5,num=100)
logMminXs = n.linspace(8,10.5,num=100)
N,M = n.meshgrid(logNxs,logMminXs)
k=0.2
z=10.9
Pk_model = build_model_interp(parm_array,delta2_array,k_array[0],z)
ki = n.abs(k-k_array[0]).argmin()
for i,alphaX in enumerate([1,2,3]):
    subplot(1,3,i+1)
    title('$\\alpha_\\mathrm{X} = $'+str(alphaX))
    im = pcolor(logNxs,logMminXs,n.log10(n.ma.masked_invalid(Pk_model[ki](N,alphaX,M))),
        vmin=.5,vmax=3)
    xlabel('log $N_\\mathrm{X}$')
    xticks([-2,-1,0,1])
    plot(n.log10(Nx_grid),n.log10(Mmin_grid),'xk')
    ylabel('log '+'$M_{\\mathrm{minX}}$')
#    if i>0:
#        yticks([])
#    else:
#        ylabel('log '+'$M_{\\mathrm{minX}}$')
#

#    grid()
tight_layout()
fig.subplots_adjust(bottom=0.32)
ax3 = fig.add_axes([0.1, 0.13, 0.85, 0.04])
cb1 = fig.colorbar(im,cax=ax3, cmap=jet,
                                orientation='horizontal')
cb1.set_label('log $\\Delta^2$')
#tight_layout()
savefig('pspec_interp_k0.2_z10.9.png')

#plot tau interpolation and maybe prior likelihood given Planck
tau_interp_model = build_tau_interp_model(parm_array)
fig =figure(figsize=(4,4))
tau_planck = 0.066
tau_planck_err = 0.012
ax = subplot(111)
#title('$\\alpha_\\mathrm{X} = $'+str(alphaX))
LOGNXS,LOGMMINXS = n.meshgrid(logNxs,logMminXs)
TAU = n.ma.masked_invalid(tau_interp_model(LOGNXS,alphaX,LOGMMINXS))
#SIGMA = (TAU-tau_planck)/tau_planck_err
#im = pcolor(logNxs,logMminXs,TAU)
print "tau range",TAU.min(),TAU.max()
tau_levels = [tau_planck,
    tau_planck+tau_planck_err/4., #added this one since all the taus are a little above tau_planck
    tau_planck+tau_planck_err/2.,
    tau_planck+tau_planck_err,
    tau_planck+tau_planck_err*3/2.,
    tau_planck+tau_planck_err*2
    ]
print "tau_levels",tau_levels
cnt = contourf(LOGNXS,LOGMMINXS,TAU,tau_levels,cmap=cm.bone,antialiased=True,
    linewidths=[2,2,4,2,4],alpha=0.5)
con = contour(LOGNXS,LOGMMINXS,TAU,tau_levels,
    linewidths=[2,2,4,2,4],colors='black')
ax.clabel(con, fontsize=12, inline=1)
xlabel('log $N_\\mathrm{X}$')
xticks([-2,-1,0,1])
xl = ax.get_xlim()
xlim([xl[0],xl[1]*.98])
yl = ax.get_ylim()
ylim([yl[0],yl[1]*.98])


ylabel('log '+'$M_{\\mathrm{minX}}$')
tight_layout()
text(-1,9.5,'0.067',fontsize=12)
savefig('tau_vs_mmin_Nx.png')
show()


