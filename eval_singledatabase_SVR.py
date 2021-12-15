import os
import warnings
import time
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
'''======================== parameters ================================'''

model_name = 'SVR'  # regression model
data_name = 'KoNViD-1k'  # dataset name CVD2014 KoNViD-1k LIVE-Qualcomm LIVE-VQC YouTube-UGC LSVQ
algo_name = 'ResNet-50'  # evaluated model
info_file = os.path.join('data', data_name+'info.mat')
feature_file = os.path.join('feature_mats', data_name, data_name+'_'+algo_name+'_feature.mat')
result_file = os.path.join('results_SVR', data_name+'_'+algo_name+'_performance.mat')

print("Evaluating algorithm {} with {} on dataset {} ...".format(algo_name,
    model_name, data_name))

'''======================== read files =============================== '''
Info = h5py.File(info_file, 'r')
Y = np.asarray(Info['scores'][0, :], dtype=np.float)

X_mat = scipy.io.loadmat(feature_file)
X = np.asarray(X_mat['feats_mat'], dtype=np.float)
# X = np.asarray(X_mat['features'], dtype=np.float)
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

'''======================== Main Body ==========================='''
model_params_all_repeats = []
PLCC_all_repeats_test = []
SRCC_all_repeats_test = []
KRCC_all_repeats_test = []
RMSE_all_repeats_test = []
PLCC_all_repeats_train = []
SRCC_all_repeats_train = []
KRCC_all_repeats_train = []
RMSE_all_repeats_train = []
# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
#
if algo_name == 'CORNIA10K' or algo_name == 'HOSA':
    C_range = [0.1, 1, 10]
    gamma_range = [0.01, 0.1, 1]
else:
    C_range = np.logspace(1, 10, 10, base=2)
    gamma_range = np.logspace(-8, 1, 10, base=2)
params_grid = dict(gamma=gamma_range, C=C_range)

# 10 random splits
for i in range(0, 10):
    print(i+1, 'th repeated 60-20-20 hold out')
    t0 = time.time()
    # parameters for each hold out
    model_params_all = []
    PLCC_all_train = []
    SRCC_all_train = []
    KRCC_all_train = []
    RMSE_all_train = []
    PLCC_all_test = []
    SRCC_all_test = []
    KRCC_all_test = []
    RMSE_all_test = []

    # Split data to test and validation sets randomly
    index = Info['index']
    index = index[:, i % index.shape[1]]
    ref_ids = Info['ref_ids'][0, :]
    index_train = index[0:int(0.6 * len(index))]
    index_valid = index[int(0.6 * len(index)):int(0.8 * len(index))]
    index_test = index[int(0.8 * len(index)):len(index)]

    index_train_real = []
    index_valid_real = []
    index_test_real = []
    for i in range(len(ref_ids)):
        if ref_ids[i] in index_train:
            index_train_real.append(i)
        if ref_ids[i] in index_valid:
            index_valid_real.append(i)
        if ref_ids[i] in index_test:
            index_test_real.append(i)

    X_train = X[index_train_real, :]
    Y_train = Y[index_train_real]
    X_valid = X[index_valid_real, :]
    Y_valid = Y[index_valid_real]
    X_test = X[index_test_real, :]
    Y_test = Y[index_test_real]

    # Standard min-max normalization of features
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    # Apply scaling
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    # SVR grid search in the TRAINING SET ONLY
    # grid search
    for C in C_range:
        for gamma in gamma_range:
            model_params_all.append((C, gamma))
            if algo_name == 'CORNIA10K' or algo_name == 'HOSA':
                model = SVR(kernel='linear', gamma=gamma, C=C)
            else:
                model = SVR(kernel='rbf', gamma=gamma, C=C)

            # Fit training set to the regression model
            model.fit(X_train, Y_train)

            # Predict MOS for the validation set
            Y_valid_pred = model.predict(X_valid)
            Y_train_pred = model.predict(X_train)

            # define 4-parameter logistic regression
            def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
                logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
                yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
                return yhat
            Y_valid = np.array(list(Y_valid), dtype=np.float)
            Y_train = np.array(list(Y_train), dtype=np.float)
            try:
                # logistic regression
                beta = [np.max(Y_valid), np.min(Y_valid), np.mean(Y_valid_pred), 0.5]
                popt, _ = curve_fit(logistic_func, Y_valid_pred, Y_valid, p0=beta, maxfev=100000000)
                Y_valid_pred_logistic = logistic_func(Y_valid_pred, *popt)
                # logistic regression
                beta = [np.max(Y_train), np.min(Y_train), np.mean(Y_train_pred), 0.5]
                popt, _ = curve_fit(logistic_func, Y_train_pred, Y_train, p0=beta, maxfev=100000000)
                Y_train_pred_logistic = logistic_func(Y_train_pred, *popt)
            except:
                raise Exception('Fitting logistic function time-out!!')
            plcc_valid_tmp = scipy.stats.pearsonr(Y_valid, Y_valid_pred_logistic)[0]
            rmse_valid_tmp = np.sqrt(mean_squared_error(Y_valid, Y_valid_pred_logistic))
            srcc_valid_tmp = scipy.stats.spearmanr(Y_valid, Y_valid_pred)[0]
            krcc_valid_tmp = scipy.stats.kendalltau(Y_valid, Y_valid_pred)[0]
            plcc_train_tmp = scipy.stats.pearsonr(Y_train, Y_train_pred_logistic)[0]
            rmse_train_tmp = np.sqrt(mean_squared_error(Y_train, Y_train_pred_logistic))
            srcc_train_tmp = scipy.stats.spearmanr(Y_train, Y_train_pred)[0]
            try:
                krcc_train_tmp = scipy.stats.kendalltau(Y_train, Y_train_pred)[0]
            except:
                krcc_train_tmp = scipy.stats.kendalltau(Y_train, Y_train_pred, method='asymptotic')[0]
            # save results
            PLCC_all_test.append(plcc_valid_tmp)
            RMSE_all_test.append(rmse_valid_tmp)
            SRCC_all_test.append(srcc_valid_tmp)
            KRCC_all_test.append(krcc_valid_tmp)
            PLCC_all_train.append(plcc_train_tmp)
            RMSE_all_train.append(rmse_train_tmp)
            SRCC_all_train.append(srcc_train_tmp)
            KRCC_all_train.append(krcc_train_tmp)

    # using the best chosen parameters to test on testing set
    param_idx = np.argmax(np.asarray(SRCC_all_test, dtype=np.float))
    C_opt, gamma_opt = model_params_all[param_idx]
    if algo_name == 'CORNIA10K' or algo_name == 'HOSA':
        model = SVR(kernel='linear', gamma=gamma_opt, C=C_opt)
    else:
        model = SVR(kernel='rbf', gamma=gamma_opt, C=C_opt)

    # Fit training set to the regression model
    model.fit(X_train, Y_train)

    # Predict MOS for the test set
    Y_test_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)
    Y_test = np.array(list(Y_test), dtype=np.float)
    Y_train = np.array(list(Y_train), dtype=np.float)
    try:
        # logistic regression
        beta = [np.max(Y_test), np.min(Y_test), np.mean(Y_test_pred), 0.5]
        popt, _ = curve_fit(logistic_func, Y_test_pred, Y_test, p0=beta, maxfev=100000000)
        Y_test_pred_logistic = logistic_func(Y_test_pred, *popt)
        # logistic regression
        beta = [np.max(Y_train), np.min(Y_train), np.mean(Y_train_pred), 0.5]
        popt, _ = curve_fit(logistic_func, Y_train_pred, Y_train, p0=beta, maxfev=100000000)
        Y_train_pred_logistic = logistic_func(Y_train_pred, *popt)
    except:
        raise Exception('Fitting logistic function time-out!!')

    plcc_test_opt = scipy.stats.pearsonr(Y_test, Y_test_pred_logistic)[0]
    rmse_test_opt = np.sqrt(mean_squared_error(Y_test, Y_test_pred_logistic))
    srcc_test_opt = scipy.stats.spearmanr(Y_test, Y_test_pred)[0]
    krcc_test_opt = scipy.stats.kendalltau(Y_test, Y_test_pred)[0]

    plcc_train_opt = scipy.stats.pearsonr(Y_train, Y_train_pred_logistic)[0]
    rmse_train_opt = np.sqrt(mean_squared_error(Y_train, Y_train_pred_logistic))
    srcc_train_opt = scipy.stats.spearmanr(Y_train, Y_train_pred)[0]
    krcc_train_opt = scipy.stats.kendalltau(Y_train, Y_train_pred)[0]

    model_params_all_repeats.append((C_opt, gamma_opt))
    SRCC_all_repeats_test.append(srcc_test_opt)
    KRCC_all_repeats_test.append(krcc_test_opt)
    PLCC_all_repeats_test.append(plcc_test_opt)
    RMSE_all_repeats_test.append(rmse_test_opt)
    SRCC_all_repeats_train.append(srcc_train_opt)
    KRCC_all_repeats_train.append(krcc_train_opt)
    PLCC_all_repeats_train.append(plcc_train_opt)
    RMSE_all_repeats_train.append(rmse_train_opt)

    # print results for each iteration
    print('======================================================')
    print('Best results in CV grid search in one split')
    print('SRCC_train: ', srcc_train_opt)
    print('KRCC_train: ', krcc_train_opt)
    print('PLCC_train: ', plcc_train_opt)
    print('RMSE_train: ', rmse_train_opt)
    print('======================================================')
    print('SRCC_test: ', srcc_test_opt)
    print('KRCC_test: ', krcc_test_opt)
    print('PLCC_test: ', plcc_test_opt)
    print('RMSE_test: ', rmse_test_opt)
    print('MODEL: ', (C_opt, gamma_opt))
    print('======================================================')
    print(' -- ' + str(time.time()-t0) + ' seconds elapsed...\n\n')

print('\n\n')
# print('======================================================')
# print('Median training results among all repeated 60-20-20 holdouts:')
# print('SRCC: ',np.median(SRCC_all_repeats_train),'( std:',np.std(SRCC_all_repeats_train),')')
# print('KRCC: ',np.median(KRCC_all_repeats_train),'( std:',np.std(KRCC_all_repeats_train),')')
# print('PLCC: ',np.median(PLCC_all_repeats_train),'( std:',np.std(PLCC_all_repeats_train),')')
# print('RMSE: ',np.median(RMSE_all_repeats_train),'( std:',np.std(RMSE_all_repeats_train),')')
# print('======================================================')
print('Median testing results among all repeated 60-20-20 holdouts:')
print('SRCC: ',np.median(SRCC_all_repeats_test),'( std:',np.std(SRCC_all_repeats_test),')')
# print('KRCC: ',np.median(KRCC_all_repeats_test),'( std:',np.std(KRCC_all_repeats_test),')')
print('PLCC: ',np.median(PLCC_all_repeats_test),'( std:',np.std(PLCC_all_repeats_test),')')
# print('RMSE: ',np.median(RMSE_all_repeats_test),'( std:',np.std(RMSE_all_repeats_test),')')
print('======================================================')
print('\n\n')
#================================================================================
# save mats
scipy.io.savemat(result_file, \
    mdict={'SRCC_train': np.asarray(SRCC_all_repeats_train,dtype=np.float), \
        'KRCC_train': np.asarray(KRCC_all_repeats_train,dtype=np.float), \
        'PLCC_train': np.asarray(PLCC_all_repeats_train,dtype=np.float), \
        'RMSE_train': np.asarray(RMSE_all_repeats_train,dtype=np.float), \
        'SRCC_test': np.asarray(SRCC_all_repeats_test,dtype=np.float), \
        'KRCC_test': np.asarray(KRCC_all_repeats_test,dtype=np.float), \
        'PLCC_test': np.asarray(PLCC_all_repeats_test,dtype=np.float), \
        'RMSE_test': np.asarray(RMSE_all_repeats_test,dtype=np.float),\
    })

a = 1
