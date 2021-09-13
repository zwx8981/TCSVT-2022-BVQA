import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

# define 4-parameter logistic regression
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

all_results = []
N = 10
method = 'QPMP'
databases = ['C'] # ['K', 'C', 'L', 'N'] # ['CQH']
database = databases[0]

for i in range(0, N):
    file_path = 'results/{}-{}-{}-{}-{}-{}-{}-{}-EXP{}.npy'.format(method,
                                      'SpatialMotion', 'mixed+spearmanr',
                                      1.0,
                                      databases, 0.0005,
                                      32, 40,
                                      i)

    test = np.load(file_path, allow_pickle=True)
    result = test.tolist()

    iter_mos = result[database]['sq']
    iter_pscore = result[database]['mq']
    # logistic regression
    beta = [np.max(iter_mos), np.min(iter_mos), np.mean(iter_pscore), 0.5]
    popt, _ = curve_fit(logistic_func, iter_pscore, iter_mos, p0=beta, maxfev=100000000)
    iter_pscore_logistic = logistic_func(iter_pscore, *popt)
    SRCC = stats.spearmanr(iter_mos, iter_pscore)[0]
    PLCC = stats.pearsonr(iter_mos, iter_pscore_logistic)[0]

    result[database]['aq'] = iter_pscore_logistic # evaluate mq but save in aq (we not use aq)
    result[database]['SROCC'] = SRCC
    result[database]['PLCC'] = PLCC

    all_results.append(result[database])

srcc = []
plcc = []
for i in range(0, N):
    srcc.append(all_results[i]['SROCC'])
    plcc.append(all_results[i]['PLCC'])
# median_srcc = np.median(srcc)
# median_plcc = np.median(plcc)
# print(median_srcc, median_plcc)
mean_srcc = np.mean(srcc)
mean_plcc = np.mean(plcc)
print(mean_srcc, mean_plcc)

a = 1
