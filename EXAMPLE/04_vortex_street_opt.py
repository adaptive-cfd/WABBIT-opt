#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOVING CYLINDERS VORTEX STREET OPTIMIZATION
@author: Philipp Krah
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('./LIB/')


import numpy as np
from numpy import exp, mod,meshgrid,pi,sin,size,cos
import matplotlib.pyplot as plt
from LIB.utils import *
import glob
from LIB.IO import read_ACM_dat
from LIB.sPOD.lib.transforms import transforms
from LIB.sPOD.lib.sPOD_tools import frame, shifted_POD, shifted_rPCA, build_all_frames
from LIB.sPOD.lib.farge_colormaps import farge_colormap_multi
from LIB.sPOD.lib.plot_utils import show_animation, save_fig
from scipy.optimize import basinhopping
import torch
import torch.nn as nn
import matplotlib
from os.path import expanduser
import os
ROOT_DIR = os.path.dirname(os.path.abspath("../README.md"))
home = expanduser("~")

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)
###############################################################################
cm = farge_colormap_multi( etalement_du_zero=0.2, limite_faible_fort=0.5)
#from sympy.physics.vector import curl
###############################################################################
def path(mu_vec):
    return (mu_vec[0] * sin(2 * pi * freq * time) + mu_vec[1] * sin(4 * pi * freq * time) + mu_vec[2] * sin(
        6 * pi * freq * time))

def dpath(mu_vec):
    return pi*freq*(2*mu_vec[0] * cos(2 * pi * freq * time) + 4*mu_vec[1] * cos(4 * pi * freq * time) + 6 * mu_vec[2] * cos(
        6 * pi * freq * time))


def my_interpolated_state(sPOD_frames, frame_amplitude_list, mu_points, Ngrid, Nt, mu_vec):
    from scipy.interpolate import griddata
    print(mu_vec)

    shift1 = np.zeros([2, Nt])
    shift2 = np.zeros([2, Nt])
    shift2[1,:] = -path(mu_vec)
    shiftsnew = [shift1, shift2]

    qtilde = 0
    for shift,frame,amplitudes in zip(shiftsnew,sPOD_frames,frame_amplitude_list):

        Modes = frame.modal_system["U"]
        VT = []
        for k in range(frame.Nmodes):
            a = griddata(mu_points.T, amplitudes[k].T, mu_vec)
            VT.append(np.squeeze(a))
        VT = np.asarray(VT)
        Q = Modes[:,:frame.Nmodes]@VT
        qframe = np.reshape(Q,[*Ngrid,1,Nt])
        trafo = transforms([*Ngrid,1,Nt], frame.trafo.domain_size, shifts=shift, dx=frame.trafo.dx,
                           use_scipy_transform=True)
        qtilde += trafo.apply(qframe)

    return qtilde


##########################################
# %% Define your DATA:
##########################################
plt.close("all")
ddir = ROOT_DIR+"/data/ai_y0_*8.2*"
frac = 4           # fraction of grid points to use
ux_list = []
uy_list = []
time_list = []
mu_list = []
Nt_sum = 0
for fpath in glob.glob(ddir):
    fpath = fpath + "/ALL.mat"
    print("reading: ", fpath)
    ux, uy, _, mask, time, Ngrid, dX, L = read_ACM_dat(fpath, sample_fraction=frac)
    ux_list.append(ux)
    uy_list.append(uy)
    time_list.append(time)
    Nt = len(time)
    Nt_sum += Nt
    mu_vec = np.asarray([float(mu) for mu in fpath.split("/")[-2].split("_")[-3:]])
    mu_list.append(mu_vec)
ux = np.concatenate(ux_list,axis=2)
uy = np.concatenate(uy_list,axis=2)
# %%

                    # Number of time intervalls
Nvar = 1# data.shape[0]                    # Number of variables
nmodes = [40,40]                              # reduction of singular values


  # number of grid points in x
data_shape = [*Ngrid,Nvar,2*Nt_sum]
               # size of time intervall
freq    = 0.01/5
Radius = 1
T = time[-1]
C_eta = 2.5e-3
x,y = (np.linspace(0, L[i]-dX[i], Ngrid[i]) for i in range(2))
dX = (x[1]-x[0],y[1]-y[0])
dt = time[1]-time[0]
[Y,X] = meshgrid(y,x)
fd = finite_diffs(Ngrid,dX)

vort= np.asarray([fd.rot(ux[...,nt],uy[...,nt]) for nt in range(np.size(ux,2))])
vort = np.moveaxis(vort,0,-1)
q = np.zeros(data_shape)
#for nvar in range(Nvar):
#for it,t in enumerate(time):
#    q[:,:,0,it] = np.array(data[0,it,::frac,::frac]).T

data = q
q = np.concatenate([ux,uy],axis=-1)
time_sum = np.concatenate([ *time_list, *time_list],axis=0)
# %%data = np.zeros(data_shape)
#for tau in range(0,Nt):
#    data[0,tau,:,:] = curl(np.squeeze(ux[0,tau,:,:]),np.squeeze(uy[0,tau,:,:]))

               # size of time intervall

shift1 = np.zeros([2,2*Nt_sum])
shift2 = np.zeros([2,2*Nt_sum])
shift1[0,:] = 0 * time_sum                      # frame 1, shift in x
shift1[1,:] = 0 * time_sum                      # frame 1, shift in y
shift2[0,:] = 0 * time_sum                      # frame 2, shift in x
y_shifts =[]
for mu_vec in mu_list:
     y_shifts.append(-path(mu_vec)) # frame 2, shift in y
shift2[1,:] = np.concatenate([*y_shifts,*y_shifts],axis=0)
# %% Create Trafo

shift_trafo_1 = transforms(data_shape,L, shifts = shift1,trafo_type="identity", dx = dX, use_scipy_transform=False )
shift_trafo_2 = transforms(data_shape,L, shifts = shift2, dx = dX, use_scipy_transform=True )
qshift1 = shift_trafo_1.reverse(q)
qshift2 = shift_trafo_2.reverse(q)
#show_animation(np.squeeze(qshift2),Xgrid=[X,Y])
qshiftreverse = shift_trafo_2.apply(qshift2)
res = q-qshiftreverse
err = np.linalg.norm(np.reshape(res,-1))/np.linalg.norm(np.reshape(q,-1))
err_time = [np.linalg.norm(np.reshape(res[...,i],-1))/np.linalg.norm(np.reshape(q[...,i],-1)) for i in range(len(time))]
print("err =  %4.4e "% err)
plt.pcolormesh(X,Y,q[...,34]-qshiftreverse[...,34])
plt.colorbar()

# %% Run shifted POD
ux_list, uy_list=None,None
ux, uy = None, None
trafos = [shift_trafo_1, shift_trafo_2]
q = np.reshape(q, [-1, 2*Nt_sum])


#ret = shifted_POD(qmat, trafos, nmodes=nmodes, eps=1e-4, Niter=100, use_rSVD=True)
[N,M]= np.shape(q)
mu0 = N * M / (4 * np.sum(np.abs(q)))*0.0005
lambd0 = 1 / np.sqrt(np.maximum(M, N))*10
#lambd0 = mu0*5e2
ret = shifted_rPCA(q, trafos, nmodes_max = np.max(nmodes)+100, eps=1e-10, Niter=100, visualize=True, use_rSVD=True, lambd=lambd0, mu=mu0)
# NmodesMax = 30
# rel_err_matrix = np.ones([30]*len(trafos))
# for r1 in range(0,NmodesMax):
#     for r2 in range(0,NmodesMax):
#         print("mode combi: [", r1,r2,"]\n")
#         ret = shifted_POD(qmat, trafos, nmodes=[r1,r2], eps=1e-4, Niter=40, use_rSVD=True)
#         print("\n\nmodes: [", r1, r2, "] error = %4.4e \n\n"%ret.rel_err_hist[-1])
#         rel_err_matrix[r1,r2] = ret.rel_err_hist[-1]


qframes, qtilde , rel_err_list = ret.frames, np.reshape(ret.data_approx,data_shape), ret.rel_err_hist
qf = [np.reshape(trafo.apply(frame.build_field()),data_shape) for trafo,frame in zip(trafos,qframes)]
E = np.reshape(ret.error_matrix,data_shape)

###########################################
# %% amplitudes
############################################
frame_amplitude_x_list = []
frame_amplitude_y_list = []
mu_vecs = np.asarray(mu_list).T
Nsamples = np.size(mu_vecs,1)
for frame in qframes:
    VT = frame.modal_system["VT"]
    S = frame.modal_system["sigma"]
    VT =  np.diag(S)@VT
    Nmodes = frame.Nmodes
    amplitudes_x = [np.reshape(VT[n, :Nt_sum], [Nsamples, Nt]).T for n in range(Nmodes)]
    amplitudes_y = [np.reshape(VT[n, Nt_sum:], [Nsamples, Nt]).T for n in range(Nmodes)]
    frame_amplitude_x_list.append(amplitudes_x)
    frame_amplitude_y_list.append(amplitudes_y)

# D=3
# fig, axs = plt.subplots(1, D, sharey=True, figsize=(18, 5), num=34)
# plt.subplots_adjust(wspace=0)
# for k, ax in enumerate(axs, start=0):
#     ax.plot(time, amplitudes_x[k][:, :D])
#     # ax.set_xlim([0, t[-1]])
#     ax.set_xticks([0, time[-1] / 2, time[-1]])
#     ax.set_title(r'${\mu}^{(' + str(k) + ')}$')
#     ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
# ax.legend([r"$i=" + str(i) + "$" for i in range(D)], loc='upper right', bbox_to_anchor=(1.5, 1))
# fig.supxlabel(r"time $t$")
# fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")


# %% Test predictions
mu_vec_test= [4,-4,4]
mu_vec_test= [8.2,-8.2,8.2]
ux_tilde = my_interpolated_state(qframes, frame_amplitude_x_list, mu_vecs, Ngrid, Nt, mu_vec_test)
uy_tilde = my_interpolated_state(qframes, frame_amplitude_y_list, mu_vecs, Ngrid, Nt, mu_vec_test)

vort_tilde= np.asarray([fd.rot(ux_tilde[...,nt],uy_tilde[...,nt]) for nt in range(Nt)])
vort_tilde = np.moveaxis(vort_tilde,0,-1)
#show_animation(np.squeeze(vort_tilde),Xgrid=[X,Y],frequency=4,vmin=-2,vmax=2)
# %%
h = 1.5*max(dX)/frac # definition of the smoothwidth of wabbit
mask2 = np.asarray([smoothstep(np.sqrt((X-L[0]/2)**2 + (Y-L[1]/2-delta)**2),Radius,h) for delta in path(mu_vec_test)])
mask2 = np.moveaxis(mask2,0,-1)
Fy= np.asarray([calculate_force(uy_tilde[...,nt] - dpath(mu_vec_test)[...,nt], (1/C_eta)*mask2[...,nt], dX) for nt in range(Nt)])
plt.plot(Fy)
print("force sum:", np.sum(Fy))
# %% optimization goal
uy_ROM_fun = lambda mu: my_interpolated_state(qframes, frame_amplitude_y_list, mu_vecs, Ngrid, Nt, mu)
give_mask = lambda mu: np.moveaxis(np.asarray([smoothstep(np.sqrt((X-L[0]/2)**2 + (Y-L[1]/2-delta)**2),Radius,h) for delta in path(mu)]), 0, -1)*1/C_eta
uy_solid = lambda mu: dpath(mu)
opt_fun = lambda mu: opt_goal_lift_drag(mu, uy_ROM_fun, give_mask, dX, uy_solid)
###########################################
# %% optimize with global optimizerN, D_out
##########################################
# mu_max = np.max(np.abs(mu_vecs))
# bounds = lambda **kwargs:  np.all(np.abs(kwargs["x_new"])<mu_max)
# bounds_i= [(-mu_max,mu_max)]*Dare
# minimizer_kwargs = {"method":"L-BFGS-B", "jac":False, "bounds": bounds_i}
# mu0 = (2*np.random.random(D)-1.0)*mu_max
# #ret_sPOD = basinhopping(opt_fun, mu0 , minimizer_kwargs=minimizer_kwargs,accept_test=bounds, seed=1,niter=5)
# #print(f"global minimum ROM:\n mu_* = {ret_sPOD.x},\n J(mu_*) = {ret_sPOD.fun}")
# #mu_star = ret_sPOD.x




# %% interpolate with NN
D=3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"
# Create input and output data
print("device: " , device)
x = torch.tensor(mu_vecs.T,dtype=torch.float32).to(device)
# output is a(mu_vec,t)
Nmodes_list = [frame.Nmodes for frame in qframes]
Nmodes_sum = np.sum(Nmodes_list)
N_samples, D_in = np.shape(mu_vecs)
# Use the nn package to define our model and loss function.
class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, Nout):
        super().__init__()
        self.Nout = Nout
        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(encoded_space_dim, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 16 * 32),
        #     nn.ReLU(True)
        # )
        #
        # self.unflatten = nn.Unflatten(dim=1,
        # unflattened_size=(32, 16))
        #
        # self.decoder_conv = nn.Sequential(
        #     nn.ConvTranspose1d(32, 64, 2,
        #     stride=2, output_padding=0),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose1d(64, 104, 2, stride=4,
        #     padding=1, output_padding=2),
        #     nn.BatchNorm1d(104),
        #     nn.ReLU(True),
        #     nn.ConvTranspose1d(104, 104, 2, stride=4,
        #     padding=1, output_padding=1)
        # )

        self.decoder_lin2 = nn.Sequential(
            nn.Linear(encoded_space_dim, 16*32),
            nn.ReLU(True),
            nn.Linear(16*32, np.prod(Nout)),
            # nn.ReLU(True),
            # nn.Linear( np.prod(Nout), np.prod(Nout))
        )

        self.unflatten2 = nn.Unflatten(dim=1,
        unflattened_size=(11,501))

    # output = s(n-1)+k-2p
    def forward(self, x):
        x = self.decoder_lin2(x)
        x = x.unflatten(1,self.Nout) # self.unflatten2(x)
        #x = self.decoder_conv(x)
        #x = torch.sigmoid(x)
        return x

    def predict_state(self,sPOD_frames, Ngrid, Nt, mu_vec):
        from scipy.interpolate import griddata
        print(mu_vec)

        shift1 = np.zeros([2, Nt])
        shift2 = np.zeros([2, Nt])
        shift2[1, :] = -path(mu_vec)
        shiftsnew = [shift1, shift2]
        Nmodes_list = [frame.Nmodes for frame in sPOD_frames]
        qtilde = 0
        mu_vec = torch.tensor(mu_vec,dtype=torch.float32)
        y_pred = np.squeeze(self.forward(mu_vec.reshape(1,3).to(device))).cpu().detach().numpy()
        frame_amplitude_list = [y_pred[ 0:Nmodes_list[0], :], y_pred[Nmodes_list[0]:, :]]
        for k,(shift, frame, amplitudes) in enumerate(zip(shiftsnew, sPOD_frames, frame_amplitude_list)):

            Modes = frame.modal_system["U"]
            VT = frame_amplitude_list[k]
            Q = Modes[:, :frame.Nmodes] @ VT
            qframe = np.reshape(Q, [*Ngrid, 1, Nt])
            trafo = transforms([*Ngrid, 1, Nt], frame.trafo.domain_size, shifts=shift, dx=frame.trafo.dx,
                               use_scipy_transform=True)
            qtilde += trafo.apply(qframe)

        return qtilde

model = Decoder(D,Nout = [Nmodes_sum,Nt]).to(device)


# def myCustomLoss(predicted_amplitudes, frames, snapshots, Nt):
#
#     batch_size = np.size(snapshots,0)
#     error = 0
#     for n in range(batch_size):
#         snap = torch.zeros_like(snapshots[...,n])
#         for frame in frames:
#             Modes = torch.tensor(frame.modal_system["U"], dtype=torch.float32)
#             VT = predicted_amplitudes[n]
#             Q = Modes[:, :frame.Nmodes] @ VT
#             qframe = np.reshape(Q.detach().numpy(), [*Ngrid, 1, Nt])
#             shift = frame.trafo.shifts_pos
#             trafo = transforms([*Ngrid, 1, Nt], frame.trafo.domain_size, shifts=shift, dx=frame.trafo.dx,
#                                use_scipy_transform=True)
#             snap += trafo.apply(qframe)
#         error += torch.norm(snap - snapshots[...,n], p = "fro")
#
#     return error/batch_size

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.

mse_loss = nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

y_truth_list = [torch.permute(torch.tensor(ampl,dtype=torch.float32),(2,0,1)).to(device) for ampl in frame_amplitude_y_list]
print("starting training")

for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)
    y_pred_list = [y_pred[:,0:Nmodes_list[0],:],y_pred[:,Nmodes_list[0]:,:]]
    # Compute and print loss.
    loss = 0
    for pred, truth in zip(y_pred_list, y_truth_list):
        loss += torch.linalg.norm(pred.flatten()- truth.flatten())**2/torch.linalg.norm(truth.flatten())**2
    #if t % 1 == 0:
    print(t, loss.item())

    # Before the backward pass use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

qtilde = model.predict_state(qframes,Ngrid,Nt,np.asarray([4.,-4.,4.]))
show_animation(np.squeeze(qtilde),Xgrid=[X,Y],frequency=4)

# %%
fpath = "/home/pkrah/develop/WABBIT-opt/data/ai_y0_4.0_-4.0_4.0/ALL.mat"
print("reading: ", fpath)
ux, uy, _,_ , _, _, _, _ = read_ACM_dat(fpath, sample_fraction=frac)
