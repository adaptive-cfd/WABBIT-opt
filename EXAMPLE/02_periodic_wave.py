#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp krah
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys

sys.path.append('./../LIB/sPOD/lib/')
import numpy as np
from numpy import exp, mod, meshgrid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sPOD_tools import shifted_rPCA, shifted_POD, build_all_frames
from transforms import transforms
from scipy.optimize import basinhopping
from scipy.fft import fft
from farge_colormaps import farge_colormap_multi

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]})

cm = farge_colormap_multi(type='vorticity', etalement_du_zero=0.02, limite_faible_fort=0.15)
import os

impath = "images/"
os.makedirs(impath, exist_ok=True)


###############################################################################

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.flip(np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8))


def my_interpolated_state(sPOD_frames, frame_amplitude_list, mu_points, D, Nx, Nt, mu_vec):
    from scipy.interpolate import griddata
    s = np.zeros(Nt)
    s[:D] = mu_vec
    print(mu_vec)
    shifts = np.asarray([fft(s).imag])
    shiftsnew = [shifts, -shifts]

    qtilde = 0
    for shift, frame, amplitudes in zip(shiftsnew, sPOD_frames, frame_amplitude_list):

        Modes = frame.modal_system["U"]
        VT = []
        for k in range(frame.Nmodes):
            a = griddata(mu_points[:D, :].T, amplitudes[k].T, mu_vec[:D], method='linear')
            VT.append(np.squeeze(a))
        VT = np.asarray(VT)
        Q = Modes[:, :frame.Nmodes] @ VT
        qframe = np.reshape(Q, [Nx, 1, 1, Nt])
        trafo = transforms([Nx, 1, 1, Nt], frame.trafo.domain_size, shifts=shift, dx=frame.trafo.dx,
                           use_scipy_transform=True)
        qtilde += trafo.apply(qframe)

    return qtilde


def interpolate_POD_states(q, mu_points, D, Nsamples, Nt, mu_vec, Nmodes):
    from scipy.interpolate import griddata

    U, S, VT = np.linalg.svd(np.squeeze(q), full_matrices=False)
    VT = np.diag(S) @ VT
    amplitudes = [np.reshape(VT[n, :], [Nsamples, Nt]).T for n in range(Nmodes)]
    VT_interp = []
    for k in range(Nmodes):
        a = griddata(mu_points[:D, :].T, amplitudes[k].T, mu_vec[:D], method='linear')
        VT_interp.append(np.squeeze(a))
    VT_interp = np.asarray(VT_interp)

    return U[:, :Nmodes] @ VT_interp


def TrainingData(sPOD_frames, mu_vecs, t):

    param = np.zeros((mu_vecs.shape[0] + 1, Nt * mu_vecs.shape[1]))
    for i in range(int(mu_vecs.shape[0]) - 1):
        for j in range(int(mu_vecs.shape[1])):
            param[i, j * Nt: (j + 1) * Nt] = mu_vecs[i, j]
            param[-1, j * Nt: (j + 1) * Nt] = t
    num_frames = 2
    S = []
    M = []
    for nf in range(num_frames):
        S.append(sPOD_frames[nf].build_field())
        M.append(param)

    np.save('frames.npy', S, allow_pickle=True)
    np.save('params.npy', M, allow_pickle=True)


if __name__ == "__main__":
    ##########################################
    # %% Define your DATA:
    ##########################################
    plt.close("all")
    Nx = 800  # number of grid points in x
    Nt = 400  # numer of time intervalls

    T = 0.5  # total time
    L = 1  # total domain size
    sigma = 0.015 * L  # standard diviation of the puls
    nmodes = 4  # reduction of singular values
    D = nmodes
    x = np.arange(-Nx // 2, Nx // 2) / Nx * L
    t = np.arange(0, Nt) / Nt * T
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    c = 1
    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T
    # mu_vecs = 0.2*L*np.eye(Nt,D)
    mu_max = 0.4 * L
    # mu_vecs = np.concatenate([mu_max*np.eye(Nt,D), -mu_max*np.eye(Nt,D)],axis=1)
    mu_vecs = np.asarray([(bin_array(i, D) * 2 - np.ones(D)) * mu_max for i in range(2 ** D)]).T
    Nsamples = np.size(mu_vecs, 1)


    def create_FOM_data(D, L, Xgrid, Tgrid, T, mu_vecs):
        from scipy.special import eval_hermite

        # gauss hermite polynomials of order n
        psi = lambda n, x: (2 ** n * np.math.factorial(n) * np.sqrt(np.pi)) ** (-0.5) * np.exp(
            -x ** 2 / 2) * eval_hermite(
            n, x)
        Nsamples = np.size(mu_vecs, 1)
        w = 0.015 * L
        Nt = np.size(Tgrid, 1)
        s = np.zeros([Nt, Nsamples])
        s[:D, :] = mu_vecs
        shifts = [np.asarray([fft(s[:, n]).imag]) for n in range(Nsamples)]
        qs1 = []
        qs2 = []
        for k in range(Nsamples):  # loop over all possible mu vectors
            q1 = np.zeros_like(Xgrid)
            q2 = np.zeros_like(Xgrid)
            for n in range(D):  # loop over all components of the vector
                q1 += np.exp(-n / 3) * mu_vecs[n, k] * np.sin(2 * np.pi * Tgrid / T * (n + 1)) * psi(n, (
                        Xgrid + 0.1 * L) / w)
                q2 += np.exp(-n / 3) * mu_vecs[n, k] * np.sin(2 * np.pi * Tgrid / T * (n + 1)) * psi(n, (
                        Xgrid - 0.1 * L) / w)

            qs1.append(q1)
            qs2.append(-q2)

        q1 = np.concatenate(qs1, axis=1)
        q2 = np.concatenate(qs2, axis=1)
        q_frames = [q1, q2]

        shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
        data_shape = [Nx, 1, 1, Nt * Nsamples]
        trafos = [transforms(data_shape, [L], shifts=shifts[0], dx=[dx], use_scipy_transform=True),
                  transforms(data_shape, [L], shifts=shifts[1], dx=[dx], use_scipy_transform=True)]

        q = 0
        for trafo, qf in zip(trafos, q_frames):
            q += trafo.apply(qf)

        return q, q1, q2, shifts, trafos


    q, q1, q2, shifts, trafos = create_FOM_data(D, L, Xgrid, Tgrid, T, mu_vecs)

    fig, axs = plt.subplots(3, D, sharey=True, sharex=True, num=33)
    plt.subplots_adjust(wspace=0)
    qmin = np.min(q)
    qmax = np.max(q)
    step = 2
    for k in range(0, Nsamples // 2, step):
        kw = k // step
        axs[0, kw].pcolormesh(q[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)
        axs[0, kw].set_title(r'${\mu}^{(' + str(k) + ')}$')
        axs[0, kw].set_yticks([0, Nx // 2, Nx])
        axs[0, kw].set_xticks([0, Nt // 2, Nt])
        axs[0, kw].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
        axs[1, kw].pcolormesh(q1[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)
        im = axs[2, kw].pcolormesh(q2[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)

    axs[0, 0].set_ylabel(r"$q$")
    axs[1, 0].set_ylabel(r"$q_1$")
    axs[2, 0].set_ylabel(r"$q_2$")
    axs[0, 0].set_xticklabels(["", r"$T/2$", r"$T$"])
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"space $x$")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.25, 0.01, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(impath + "1Dwaves.png", dpi=600, transparent=True)

    mu = np.prod(np.shape(q)) / (4 * np.sum(np.abs(q))) * 0.001
    ret = shifted_rPCA(q, trafos, nmodes_max=np.max(nmodes) + 10, eps=1e-16, Niter=500, use_rSVD=True, mu=mu, lambd=0.1)
    # ret  = shifted_POD(q, trafos, nmodes, eps=1e-16, Niter=400, use_rSVD = False)
    sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist

    # Save the snapshot and parameter matrices for training.
    TrainingData(sPOD_frames, mu_vecs, t)
    S = np.load('frames.npy', allow_pickle=True)
    M = np.load('params.npy', allow_pickle=True)

    ###########################################
    # %% results sPOD frames
    ##########################################
    # the result is a list of the decomposed field.
    # each element of the list contains a frame of the decomposition.<
    # If you want to plot the k-th frame use:
    # 1. frame
    fig, axs = plt.subplots(1, 3, num=10, sharey=True, figsize=(10, 3))
    k_frame = 0

    axs[0].pcolormesh(sPOD_frames[k_frame].build_field()[:, :Nt], cmap=cm, vmin=qmin, vmax=qmax, )
    # fig.suptitle("sPOD Frames")
    axs[0].set_ylabel(r'$i=1,\dots,M$')
    axs[0].set_xlabel(r'$j=1,\dots,ND$')
    axs[0].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
    # 2. frame
    k_frame = 1
    axs[1].pcolormesh(sPOD_frames[k_frame].build_field()[:, :Nt], cmap=cm, vmin=qmin, vmax=qmax, )
    # axs[1].set_ylabel(r'$i=1,\dots,M$')
    axs[1].set_xlabel(r'$j=1,\dots,ND$')
    axs[1].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
    # approximation
    k_frame = 1
    axs[2].pcolormesh(qtilde[:, :Nt], cmap=cm, vmin=qmin, vmax=qmax)
    # axs[2].set_ylabel(r'$i=1,\dots,M$')
    axs[2].set_xlabel(r'$j=1,\dots,ND$')
    axs[2].set_title(r"$\tilde{Q}" + "_{i,j}$")
    plt.tight_layout()
    fig.savefig(impath + "1Dwaves_frames.png", dpi=600, transparent=True)
    # by default this will plot the field in the first component
    # of your field list (here: density)

    ###########################################
    # relative error
    ##########################################

    plt.figure(5)
    plt.rcParams.update({"font.size": 21})
    plt.semilogy(rel_err)
    plt.title("relative error")
    plt.ylabel(r"$\frac{||X - \tilde{X}_i||_2}{||X||_2}$")
    plt.xlabel(r"iteration $i$")

    # %% offline error
    fig = plt.figure(4)
    U, S, VT = np.linalg.svd(np.squeeze(q), full_matrices=False)
    plt.semilogy(1 - np.cumsum(S) / np.sum(S), '--*', label=r"POD")

    qnorm = norm(np.squeeze(q), ord="fro")
    rel_err_list = []
    DoF_list = []
    for r in range(1, np.max(nmodes) + 2):
        qtilde = build_all_frames(sPOD_frames, trafos, ranks=r)
        rel_err = norm(q - qtilde, ord="fro") / qnorm
        rel_err_list.append(rel_err)
        DoF_list.append(r)

    plt.semilogy(2 * np.asarray(DoF_list), rel_err_list, 'o', label=r"sPOD")
    plt.xlim([-1, 200])
    plt.legend()
    plt.xlabel(r"$r$ DoFs")
    plt.ylabel(r"rel err")
    plt.tight_layout()
    fig.savefig(impath + "1Dwaves_frames_offline_err.png", dpi=600, transparent=True)

    ###########################################
    # %% amplitudes
    ############################################
    frame_amplitude_list = []
    for frame in sPOD_frames:
        VT = frame.modal_system["VT"]
        S = frame.modal_system["sigma"]
        VT = np.diag(S) @ VT
        Nmodes = frame.Nmodes
        amplitudes = [np.reshape(VT[n, :], [Nsamples, Nt]).T for n in range(Nmodes)]
        frame_amplitude_list.append(amplitudes)

    fig, axs = plt.subplots(1, D, sharey=True, figsize=(18, 5), num=34)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, amplitudes[k][:, :D])
        # ax.set_xlim([0, t[-1]])
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${\mu}^{(' + str(k) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
    ax.legend([r"$i=" + str(i) + "$" for i in range(D)], loc='upper right', bbox_to_anchor=(1.5, 1))
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    # fig.tight_layout()

    #############################
    # %% online error
    #############################

    fig = plt.figure(6)
    Ntest = 16
    mu_test = (2 * np.random.random([D, Ntest]) - 1.0) * mu_max  # [0
    Qtest, _, _, _, _ = create_FOM_data(D, L, Xgrid, Tgrid, T, mu_test)

    qnorm = norm(Qtest, ord="fro")
    rel_err_list = []
    DoF_list = []
    rel_err = np.ones([Ntest, np.max(nmodes) + 2])
    rel_err_POD = np.ones([Ntest, np.max(nmodes) + 2])
    for r in range(1, np.max(nmodes) + 2):
        for i in range(Ntest):
            sPOD_frames[0].Nmodes = r
            sPOD_frames[1].Nmodes = r
            qtilde = my_interpolated_state(sPOD_frames, frame_amplitude_list, mu_vecs, D, Nx, Nt, mu_test[:D, i])
            qnorm = norm(Qtest[:, i * Nt:Nt * (i + 1)], ord="fro")
            rel_err[i, r] = norm(Qtest[:, i * Nt:Nt * (i + 1)] - np.squeeze(qtilde), ord="fro") / qnorm
            qtilde = interpolate_POD_states(q, mu_vecs, D, Nsamples, Nt, mu_test[:D, i], 2 * r)
            rel_err_POD[i, r] = norm(Qtest[:, i * Nt:Nt * (i + 1)] - np.squeeze(qtilde), ord="fro") / qnorm

    dofs = np.arange(0, np.max(nmodes) + 2)
    err_var = np.var(rel_err_POD, axis=0)
    err_mean = np.mean(rel_err_POD, axis=0)
    plt.errorbar(2 * dofs, err_mean, marker='x', linestyle='', yerr=err_var, label=r"POD")
    err_var = np.var(rel_err, axis=0)
    err_mean = np.mean(rel_err, axis=0)
    plt.errorbar(2 * dofs, err_mean, marker='o', linestyle='', yerr=err_var, label=r"sPOD")

    plt.legend()
    plt.xlabel(r"$r$ DoFs")
    plt.ylabel(r"rel err")
    plt.tight_layout()
    fig.savefig(impath + "1Dwaves_frames_online_err.png", dpi=600, transparent=True)

    plt.rcParams.update({"font.size": 16})

    fig, axs = plt.subplots(1, 3, num=11, sharey=True, figsize=(10, 3))

    qtilde = my_interpolated_state(sPOD_frames, frame_amplitude_list, mu_vecs, D, Nx, Nt, mu_test[:D, 0])
    axs[0].pcolormesh(np.squeeze(qtilde), cmap=cm, vmin=qmin, vmax=qmax, )
    axs[0].set_ylabel(r'space $x$')
    axs[0].set_xlabel(r'time $t$')
    axs[0].set_title(r"sPOD")
    qtilde = interpolate_POD_states(q, mu_vecs, D, Nsamples, Nt, mu_test[:D, 0], 100)
    axs[1].pcolormesh(np.squeeze(qtilde), cmap=cm, vmin=qmin, vmax=qmax, )
    # axs[1].set_ylabel(r'$i=1,\dots,M$')
    axs[1].set_xlabel(r'time $t$')
    axs[1].set_title(r"POD")
    # approximation
    axs[2].pcolormesh(Qtest[:, :Nt], cmap=cm, vmin=qmin, vmax=qmax)
    # axs[2].set_ylabel(r'$i=1,\dots,M$')
    axs[2].set_xlabel(r'time $t$')
    axs[2].set_title(r"Data")
    plt.tight_layout()
    fig.savefig(impath + "1Dwaves_online_prediction.png", dpi=600, transparent=True)

    fig.savefig(impath + "1Dwaves_frames_online_err.png", dpi=600, transparent=True)

    ###########################################
    # %% define optimization goal
    ##########################################
    Wt = np.zeros([Nt, 1])
    Wt[Nt // 4:3 * Nt // 4] = 1
    Wx = np.tanh((x - 0.4 * L) / 50 / dx) * 0.5 + 0.5

    mu_goal = np.zeros([1, D]).T
    mu_goal[:4, 0] = (2 * np.random.random(4) - 1.0) * mu_max / 2  # [0
    Qgoal, _, _, _, _ = create_FOM_data(D, L, Xgrid, Tgrid, T, mu_goal)
    Qgoal = np.squeeze(Qgoal)
    # opt_goal = lambda Qmat: norm(Qmat-Qgoal,ord='fro')**2*dx*dt
    opt_goal = lambda Qmat: -dx * dt * norm(
        np.abs(Qmat) * np.exp(-0.5 * (np.abs(Xgrid) - 0.3 * L) ** 2 / (0.02 * L) ** 2)) ** 2
    opt_fun_sPOD = lambda mu: opt_goal(
        np.reshape(my_interpolated_state(sPOD_frames, frame_amplitude_list, mu_vecs, D, Nx, Nt, mu), [Nx, Nt]))
    opt_fun_FOM = lambda mu: opt_goal(
        np.reshape(create_FOM_data(D, L, Xgrid, Tgrid, T, np.reshape(mu, [D, 1]))[0], [Nx, Nt]))
    ###########################################
    # %% optimize with global optimizer
    ##########################################

    bounds = lambda **kwargs: np.all(np.abs(kwargs["x_new"]) < mu_max)
    bounds_i = [(-mu_max, mu_max)] * D
    minimizer_kwargs = {"method": "L-BFGS-B", "jac": False, "bounds": bounds_i}
    mu0 = (2 * np.random.random(4) - 1.0) * mu_max
    ret_sPOD = basinhopping(opt_fun_sPOD, mu0, minimizer_kwargs=minimizer_kwargs, accept_test=bounds, seed=1, niter=5)
    print(f"global minimum ROM:\n mu_* = {ret_sPOD.x},\n J(mu_*) = {ret_sPOD.fun}")
    ret_FOM = basinhopping(opt_fun_FOM, mu0, minimizer_kwargs=minimizer_kwargs, accept_test=bounds, seed=1, niter=5)
    print(f"global minimum FOM:\n mu_* = {ret_FOM.x},\n J(mu_*) = {ret_FOM.fun}")

    ###########################################
    # %% plot results of optimization
    ##########################################
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 3))  #
    plt.rcParams.update({"font.size": 16})
    qtilde_FOM = np.squeeze(create_FOM_data(D, L, Xgrid, Tgrid, T, np.reshape(ret_FOM.x, [D, 1]))[0])
    qmin = np.min(qtilde_FOM)
    qmax = np.max(qtilde_FOM)
    ax[0].pcolormesh(qtilde_FOM, cmap=cm, vmin=qmin, vmax=qmax)
    ax[0].set_title(r"FOM")
    ax[0].set_yticks([0, Nx // 2, Nx])
    ax[0].set_xticks([0, Nt // 2, Nt])
    ax[0].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
    ax[0].set_xticklabels([r"$0$", r"T/2", r"$T$"])
    ax[0].set_xlabel(r"time $t$")
    ax[0].set_ylabel(r"$x$")

    qtilde_ROM = np.squeeze(my_interpolated_state(sPOD_frames, frame_amplitude_list, mu_vecs, D, Nx, Nt, ret_sPOD.x))
    ax[1].pcolormesh(qtilde_ROM, cmap=cm, vmin=qmin, vmax=qmax)
    ax[1].set_title(r"ROM")
    ax[1].set_yticks([0, Nx // 2, Nx])
    ax[1].set_xticks([0, Nt // 2, Nt])
    ax[1].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
    ax[1].set_xticklabels([r"$0$", r"T/2", r"$T$"])
    ax[1].set_xlabel(r"time $t$")
    ax[1].set_ylabel(r"$x$")

    ax[2].pcolormesh(np.exp(-0.5 * (np.abs(Xgrid) - 0.3 * L) ** 2 / (0.02 * L) ** 2), cmap=cm, vmin=qmin, vmax=qmax)
    ax[2].set_title(r"weight: $e^{-\frac{(|x|-x_0)^2}{2d^2}}$")
    ax[2].set_yticks([0, Nx // 2, Nx])
    ax[2].set_xticks([0, Nt // 2, Nt])
    ax[2].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
    ax[2].set_xticklabels([r"$0$", r"T/2", r"$T$"])
    ax[2].set_xlabel(r"time $t$")
    ax[2].set_ylabel(r"$x$")
    plt.tight_layout()
    fig.savefig(impath + "1Dwaves_opt.png", dpi=600, transparent=True)
