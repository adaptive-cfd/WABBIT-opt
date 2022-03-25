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
sys.path.append('./../LIB/MLbasedModelReduction/POD_DL_ROM_LIB/')
sys.path.append('./../LIB/sPOD/lib/')
import pathlib
import numpy as np
from numpy import exp, mod, meshgrid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.utils import extmath
import os
import numpy as np
from numpy import exp, mod, meshgrid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from transforms import transforms
from scipy.fft import fft

from TrainingFramework import TrainingFramework
from TestingFramework import TestingFramework

datapath = "MLdata/"
os.makedirs(datapath, exist_ok=True)



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


def network_data(mu_vecs, mu_vecs_test, qtrain1, qtrain2, qtest1, qtest2, t):
    param_train = np.zeros((mu_vecs.shape[0] + 1, Nt * mu_vecs.shape[1]))
    param_test = np.zeros((mu_vecs_test.shape[0] + 1, Nt * mu_vecs_test.shape[1]))
    for i in range(int(mu_vecs.shape[0])):
        for j in range(int(mu_vecs.shape[1])):
            param_train[i, j * Nt: (j + 1) * Nt] = mu_vecs[i, j]
            param_train[-1, j * Nt: (j + 1) * Nt] = t
    for i in range(int(mu_vecs_test.shape[0])):
        for j in range(int(mu_vecs_test.shape[1])):
            param_test[i, j * Nt: (j + 1) * Nt] = mu_vecs_test[i, j]
            param_test[-1, j * Nt: (j + 1) * Nt] = t

    nf = 0
    M_train = []
    M_test = []
    for frame in range(2):
        M_train.append(param_train)
        M_test.append(param_test)
        nf = nf + 1

    qtest_frame_wise = [qtest1, qtest2]
    qtrain_frame_wise = [qtrain1, qtrain2]

    np.save(datapath + "params_train.npy", M_train, allow_pickle=True)
    np.save(datapath + "params_test.npy", M_test, allow_pickle=True)
    np.save(datapath + "snapshot_test_frame.npy", qtest_frame_wise, allow_pickle=True)
    np.save(datapath + "snapshot_train_frame.npy", qtrain_frame_wise, allow_pickle=True)

    pass


if __name__ == "__main__":
    ##########################################
    # %% Define your DATA:
    ##########################################
    # plt.close("all")
    # Nx = 800  # number of grid points in x
    # Nt = 400  # numer of time intervalls
    #
    # T = 0.5  # total time
    # L = 1  # total domain size
    # sigma = 0.015 * L  # standard diviation of the puls
    # nmodes = 4  # reduction of singular values
    # D = nmodes
    # x = np.arange(-Nx // 2, Nx // 2) / Nx * L
    # t = np.arange(0, Nt) / Nt * T
    # dx = x[1] - x[0]
    # dt = t[1] - t[0]
    # c = 1
    # [Xgrid, Tgrid] = meshgrid(x, t)
    # Xgrid = Xgrid.T
    # Tgrid = Tgrid.T
    # # mu_vecs = 0.2*L*np.eye(Nt,D)
    # mu_max = 0.4 * L
    # # mu_vecs = np.concatenate([mu_max*np.eye(Nt,D), -mu_max*np.eye(Nt,D)],axis=1)
    # mu_vecs = np.asarray([(bin_array(i, D) * 2 - np.ones(D)) * mu_max for i in range(2 ** D)]).T
    # Nsamples = np.size(mu_vecs, 1)
    #
    #
    # def create_FOM_data(D, L, Xgrid, Tgrid, T, mu_vecs):
    #     from scipy.special import eval_hermite
    #
    #     # gauss hermite polynomials of order n
    #     psi = lambda n, x: (2 ** n * np.math.factorial(n) * np.sqrt(np.pi)) ** (-0.5) * np.exp(
    #         -x ** 2 / 2) * eval_hermite(
    #         n, x)
    #     Nsamples = np.size(mu_vecs, 1)
    #     w = 0.015 * L
    #     Nt = np.size(Tgrid, 1)
    #     s = np.zeros([Nt, Nsamples])
    #     s[:4, :] = mu_vecs
    #     shifts = [np.asarray([fft(s[:, n]).imag]) for n in range(Nsamples)]
    #     qs1 = []
    #     qs2 = []
    #     for k in range(Nsamples):  # loop over all possible mu vectors
    #         q1 = np.zeros_like(Xgrid)
    #         q2 = np.zeros_like(Xgrid)
    #         print(k)
    #         for n in range(D):  # loop over all components of the vector
    #             q1 += np.exp(-n / 3) * np.sin(2 * np.pi * Tgrid / T * (n + 1)) * psi(n, (
    #                     Xgrid + 0.1 * L) / w)
    #             q2 += np.exp(-n / 3) * np.sin(2 * np.pi * Tgrid / T * (n + 1)) * psi(n, (
    #                     Xgrid - 0.1 * L) / w)
    #
    #         qs1.append(q1)
    #         qs2.append(-q2)
    #
    #     q1 = np.concatenate(qs1, axis=1)
    #     q2 = np.concatenate(qs2, axis=1)
    #     q_frames = [q1, q2]
    #
    #     shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
    #     data_shape = [Nx, 1, 1, Nt * Nsamples]
    #     trafos = [transforms(data_shape, [L], shifts=shifts[0], dx=[dx], use_scipy_transform=True),
    #               transforms(data_shape, [L], shifts=shifts[1], dx=[dx], use_scipy_transform=True)]
    #
    #     q = 0
    #     for trafo, qf in zip(trafos, q_frames):
    #         q += trafo.apply(qf)
    #
    #     return q, q1, q2, shifts, trafos
    #
    #
    # q, q1, q2, shifts, trafos = create_FOM_data(16, L, Xgrid, Tgrid, T, mu_vecs)
    #
    # Ntest = 16
    # mu_test = (2 * np.random.random([D, Ntest]) - 1.0) * mu_max  # [0
    # Qtest, qtest1, qtest2, _, _ = create_FOM_data(16, L, Xgrid, Tgrid, T, mu_test)
    #
    # network_data(mu_vecs, mu_test, q1, q2, qtest1, qtest2, t)
    #
    # sys.exit()

    #############################################################
    ## ML section
    log_dir = "MLdata/"
    TrainingSnapShotMatrix = np.load(log_dir + 'snapshot_train_frame.npy', allow_pickle=True)
    TestingSnapShotMatrix = np.load(log_dir + 'snapshot_test_frame.npy', allow_pickle=True)
    TrainingParameterMatrix = np.load(log_dir + 'params_train.npy', allow_pickle=True)
    TestingParameterMatrix = np.load(log_dir + 'params_test.npy', allow_pickle=True)

    # Parameters needed for the training and validation of the framework
    params = {
        'FOM': True,  # This switch is true for full order model input and false for only time amplitude matrix
        'snapshot_train': TrainingSnapShotMatrix[0],
        'snapshot_test': TestingSnapShotMatrix[0],
        'time_amplitude_train': None,
        'time_amplitude_test': None,
        'parameter_train': TrainingParameterMatrix[0],
        'parameter_test': TestingParameterMatrix[0],
        'num_parameters': int(TrainingParameterMatrix[0].shape[0]),  # n_mu + 1
        'num_time_steps': 400,  # N_t
        'num_samples': int(TrainingSnapShotMatrix[0].shape[1]),  # N_train x N_t
        'num_test_samples': int(TestingSnapShotMatrix[0].shape[1]),  # N_test x N_t
        'batch_size': 400,
        'num_early_stop': 1500,  # Number of epochs for the early stopping
        'scaling': True,  # true if the data should be scaled
        'perform_svd': 'randomized',  # '', 'normal', 'randomized'
        'learning_rate': 0.0005,  # eta  0.001
        'full_order_model_dimension': int(TrainingSnapShotMatrix[0].shape[0]),  # N_h
        'reduced_order_model_dimension': 16,  # N
        'encoded_dimension': 7,  # dimension of the system after the encoder
        'num_dimension': 1,  # Number of dimensions (d)
        'omega_h': 0.8,
        'omega_N': 0.2,
        'typeConv': '1D'
    }

    # POD_DL_ROM = TrainingFramework(params, split=0.67)
    # POD_DL_ROM.training(epochs=50, save_every=10, print_every=10)
    #
    # sys.exit()

    testing_method = 'weight_based'
    if testing_method == 'model_based':
        log_folder_base = 'training_results_local/'
        num_frame_models = 1
        log_folder_trained_model = []
        for num_frame in range(num_frame_models):
            f = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-(num_frame + 1)]
            log_folder_trained_model.append(f)
        log_folder_trained_model.reverse()

        time_amplitudes_predicted = []
        # Testing for each frame
        for frame, folder_name in enumerate(log_folder_trained_model):
            # Testing for collection of snapshots
            test_model = TestingFramework(params)
            test_model.testing(log_folder_trained_model=str(folder_name), testing_method='model_based')
            time_amplitudes_predicted.append(test_model.time_amplitude_test_output)
    else:
        log_folder_base = 'training_results_local/'
        num_frame_models = 1
        log_folder_trained_model = []
        for num_frame in range(num_frame_models):
            f = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-(num_frame + 1)]
            log_folder_trained_model.append(f)
        log_folder_trained_model.reverse()

        time_amplitudes_predicted = []
        # Testing for each frame
        for frame, folder_name in enumerate(log_folder_trained_model):
            # Testing for collection of snapshots
            test_model = TestingFramework(params)
            test_model.testing(log_folder_trained_model=str(folder_name), testing_method='weight_based')
            time_amplitudes_predicted.append(test_model.time_amplitude_test_output)

    # Plot and testing
    Nt = params['num_time_steps']
    num_instances = params['snapshot_test'].shape[1] // Nt

    N_h = params['full_order_model_dimension']
    N = params['reduced_order_model_dimension']
    num_dim = params['num_dimension']

    NN_err = np.zeros((num_instances, 1))
    POD_err = np.zeros((num_instances, 1))
    t_err = np.zeros((num_instances, 1))
    SnapMat_NN = np.zeros_like(params['snapshot_test'])
    SnapMat_POD = np.zeros_like(params['snapshot_test'])
    time_amplitudes = np.zeros((num_dim * N, Nt))

    U, _, _ = extmath.randomized_svd(params['snapshot_train'],
                                     n_components=params['reduced_order_model_dimension'],
                                     transpose=False,
                                     flip_sign=False, random_state=123)

    for fr in range(num_frame_models):
        for i in range(num_instances):
            for j in range(params['num_dimension']):
                SnapMat_NN[j * N_h:(j + 1) * N_h, i * Nt:(i + 1) * Nt] = \
                    np.matmul(U[j * N_h:(j + 1) * N_h, :],
                              time_amplitudes_predicted[fr][j * N:(j + 1) * N, i * Nt:(i + 1) * Nt])
                time_amplitudes[j * N:(j + 1) * N, :] = np.matmul(
                    U.transpose()[:, j * N_h:(j + 1) * N_h],
                    params['snapshot_test'][j * N_h:(j + 1) * N_h, i * Nt:(i + 1) * Nt])
                SnapMat_POD[j * N_h:(j + 1) * N_h, i * Nt:(i + 1) * Nt] = \
                    np.matmul(U[j * N_h:(j + 1) * N_h, :],
                              time_amplitudes[j * N:(j + 1) * N, :])

            num = np.sqrt(np.mean(np.linalg.norm(
                time_amplitudes[:, :] -
                time_amplitudes_predicted[fr][:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                time_amplitudes[:, :], 2, axis=1) ** 2))
            t_err[i] = num / den

            num = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt] -
                SnapMat_NN[:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            NN_err[i] = num / den

            num = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt] -
                SnapMat_POD[:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(
                params['snapshot_test'][:, i * Nt:(i + 1) * Nt], 2, axis=1) ** 2))
            POD_err[i] = num / den

            print('Relative time amplitude error indicator: {0}'.format(t_err[i]))
            print('Relative NN reconstruction error indicator: {0}'.format(NN_err[i]))
            print('Relative POD reconstruction error indicator: {0}'.format(POD_err[i]))
            print('\n')

            X = np.linspace(0, 400, 400)
            plt.plot(X, time_amplitudes[0, :])
            plt.plot(X, time_amplitudes_predicted[fr][0, i * Nt:(i + 1) * Nt])
            plt.show()

        print(np.mean(t_err), np.mean(NN_err), np.mean(POD_err))


