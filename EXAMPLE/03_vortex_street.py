"""
Note this routine generates Nsamples folder in the data directory, clones wabbit into them and
runs wabbit for the different parameters. You may want to change the MPI_COMMAND for cluster or
desktop settings.
"""

# uncomment to execute script:
# from LIB.run_FOM


"""
Load the statevector and mask function
"""
import sys
sys.path.append('./../LIB/')
from LIB.ROM_utils import my_interpolated_state
from numpy import concatenate
from LIB.IO import read_ACM_dat
from LIB.IO import load_trajectories
from IPython.display import HTML
import numpy as np
from numpy import reshape, size, concatenate, shape, meshgrid
from LIB.sPOD.lib.plot_utils import show_animation, save_fig
from LIB.utils import finite_diffs
from farge_colormaps import farge_colormap_multi
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

cm = farge_colormap_multi(type='vorticity', etalement_du_zero=0.02,limite_faible_fort=0.15)

data_path = "../data/"
frac = 2
time_frac=20
_, ux1, dX, time_list1, mu_vec_list1, L = load_trajectories(data_path,["ux","uy","dx","time","mu","domain_size"],
                                                               params_id_regex = "ai_y0_8.2*", sample_fraction=frac, time_sample_fraction=time_frac)

_, ux2, dX, time_list2, mu_vec_list2, L = load_trajectories(data_path,["ux","uy","dx","time","mu","domain_size"],
                                                               params_id_regex = "ai_y0_4.0*", sample_fraction=frac, time_sample_fraction=time_frac)
ux =concatenate([*ux1,*ux2],axis=2)
ux2,ux1 = None, None
#uy = concatenate([*uy1,*uy2],axis=2)
uy1, uy2 = None, None
time_list = [*time_list1,*time_list2]
mu_vec_list = [*mu_vec_list1, *mu_vec_list2]
time = time_list[0]
time_joint = concatenate(time_list)

# %%

import matplotlib.pyplot as plt
# quantities
Ngrid = shape(ux[...,0])
Nt = len(time)
matr = lambda dat: np.reshape(dat,[-1,np.size(dat,-1)])
freq0 = 0.2e-2
x,y = (np.linspace(0, L[i]-dX[i], Ngrid[i]) for i in range(2))
[Y,X] = meshgrid(y,x)
# build snapshotmatrix
#uy[:,Ngrid[1]//2,:]=10
#ux[:,Ngrid[1]//2,:]=10
Q = np.concatenate([matr(ux)],axis=0)
ux = None
time_joint = concatenate([*time_list])
Nsnapshots = size(Q,-1)
Nsamples = Nsnapshots//Nt//2 # devided by two since Q has ux, and uy in columns not in rows
print("(Nx, Ny) =", Ngrid )
print("Timesamples =", Nt )
print("Nsamples = ", Nsamples )
print("Nsnapshots = ", Nsnapshots )

# fd = finite_diffs(Ngrid,dX)
# vort= np.asarray([fd.rot(ux[...,nt],uy[...,nt]) for nt in range(np.size(ux,2))])
# vort = np.moveaxis(vort,0,-1)
#show_animation(np.squeeze(vort),Xgrid=[X,Y],frequency=4,vmin=-2,vmax=2)
# %%
from numpy import sin,cos,pi

def path(mu_vec, time, freq):
    d2 = 0
    for k in range(len(mu_vec)):
        d2 += mu_vec[k] * sin(2 *(k+1)* pi * freq * time)
    return d2
def dpath(mu_vec, time , freq):
    d2dot = 0
    for k in range(len(mu_vec)):
        d2dot += 2 *(k+1)* pi * freq * mu_vec[k] * cos(2 *(k+1)* pi * freq * time)
    return d2dot

def give_shift(time,x, mu_vec, freq):
    shift = np.zeros([len(x),len(time)])
    for it,t in enumerate(time):
        shift[...,it] = path(mu_vec,np.heaviside(x,0)*(x)-t,freq)
    return shift
shift1 = np.zeros([2,np.prod(Ngrid),Nsnapshots])
shift2 = np.zeros([2,np.prod(Ngrid),Nsnapshots])
# shift1 = np.zeros([2,Nsnapshots])
# shift2 = np.zeros([2,Nsnapshots])

# shift1[0,:] = 0 * time_joint                      # frame 1, shift in x
# shift1[1,:] = 0 * time_joint                      # frame 1, shift in y
# shift2[0,:] = 0 * time_joint                      # frame 2, shift in x
y_shifts =[]
dy_shifts = []
for mu_vec in mu_vec_list:
     y_shifts.append(give_shift(time,X.flatten()-L[0]/2, mu_vec, freq0))
     #y_shifts.append(-path(mu_vec, time, freq0)) # frame 2, shift in y
     dy_shifts.append(dpath(mu_vec, time, freq0))  # frame 2, shift in y
dy_shifts = np.concatenate([*dy_shifts],axis=-1)
shift2[1,...] = np.concatenate([*y_shifts],axis=-1)


# %%
from numpy.linalg import norm
from LIB.sPOD.lib.transforms import transforms

class ACM_transforms(transforms):
    def __init__(self, data_shape, domain_size, trafo_type="shift", shifts = None, dshifts =None,\
                 dx = None, rotations=None, rotation_center = None, use_scipy_transform = False):
        super().__init__(data_shape, domain_size, trafo_type, shifts , dx,
                         rotations, rotation_center , use_scipy_transform )
        self.dshifts = dshifts

    def apply(self, frame_field):
        lab_field = super().apply(frame_field) #+ self.dshifts
        return lab_field

    def reverse(self, lab_field):
        frame_field = super().reverse(lab_field) #- self.dshifts
        return frame_field



data_shape = [*Ngrid,1,Nsnapshots]
# first trafo is the identity
shift_trafo_1 = transforms(data_shape,L, shifts = shift1,trafo_type="identity", dx = dX, use_scipy_transform=False )
# second is the shift
shift_trafo_2 = ACM_transforms(data_shape,L, shifts = shift2, dshifts=dy_shifts, dx = dX, use_scipy_transform=False )

# print( "rel interpolation error: %4.4e"%(norm(Q - shift_trafo_2.apply(shift_trafo_2.reverse(Q)))/norm(Q)))
# Qt = shift_trafo_2.reverse(Q)
# Qt = shift_trafo_2.apply(Qt)
# interp_time_err = [norm(qcol - qtcol)/norm(qcol)  for (qcol,qtcol) in zip(Q.T,Qt.T)]
# %%
from LIB.sPOD.lib.sPOD_tools import shifted_POD, shifted_rPCA

nmodes = [100,85]
trafos = [shift_trafo_1, shift_trafo_2]
M = np.prod(Ngrid)
mu0 = M * Nsnapshots / (4 * np.sum(np.abs(Q)))*0.0005
lambd0 = 1 / np.sqrt(np.maximum(M, Nsnapshots))*10
#lambd0 = mu0*5e2
#ret = shifted_rPCA(Q, trafos, nmodes_max = np.max(nmodes)+20, eps=1e-10, Niter=100, visualize=True, use_rSVD=True, lambd=lambd0, mu=mu0)
ret = shifted_POD(Q, trafos, nmodes, eps=1e-4, Niter=300, use_rSVD=True)
qframes, qtilde , rel_err_list = ret.frames, np.reshape(ret.data_approx,data_shape), ret.rel_err_hist
#qf = [np.reshape(trafo.apply(frame.build_field()),data_shape) for trafo,frame in zip(trafos,ret.frames)]


###########################################
# %% amplitudes
############################################
frame_amplitude_list= []
mu_vecs = np.asarray(mu_vec_list).T
Nsamples = np.size(mu_vecs,1)
Nt = len(time)
for frame in qframes:
    VT = frame.modal_system["VT"]
    S = frame.modal_system["sigma"]
    VT =  np.diag(S)@VT
    Nmodes = frame.Nmodes
    frame_amplitude_list.append(VT)

# %%
#mu_vec_test= [4,4,4]
mu_vec_test= [2,2,2]
Nt = len(time)

shift1 = np.zeros([2,np.prod(Ngrid), Nt])
shift2 = np.zeros([2,np.prod(Ngrid), Nt])
shift2[1, ...] = give_shift(time,X.flatten()-L[0]/2, mu_vec_test, freq0)
shiftsnew = [shift1, shift2]
ux_tilde = my_interpolated_state(qframes, frame_amplitude_list, mu_vecs, Ngrid, time, mu_vec_test, shiftsnew )

# %%
fpath = "/home/pkrah/develop/WABBIT-opt/data/ai_y0_"+"_".join(["%2.1f"%mu for mu in mu_vec_test])+"/ALL.mat"
print("reading: ", fpath)
_, ux, _,_ , _, _, _, _ = read_ACM_dat(fpath, sample_fraction=frac, time_sample_fraction=time_frac)

error_online = np.linalg.norm(np.reshape(ux-np.squeeze(ux_tilde),[1,-1]),ord="fro")/np.linalg.norm(np.reshape(ux,[1,-1]),ord="fro")
print("online-err: ", error_online)





