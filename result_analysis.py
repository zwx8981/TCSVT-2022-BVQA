import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from argparse import ArgumentParser

# define 4-parameter logistic regression
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

if __name__ == "__main__":
    parser = ArgumentParser(description='Result Analysis for Quality Assessment of In-the-Wild Videos')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--model', default='QPMP', type=str,
                        help='model name (default: QPMP)')
    parser.add_argument('--loss', default='plcc+srcc', type=str,
                        help='loss type (default: plcc+srcc)')
    parser.add_argument('--feature_extractor', default='SpatialMotion', type=str,
                        help='feature_extractor backbone (default: ResNet-50)')
    parser.add_argument('--trained_datasets', nargs='+', type=str, default=['C'],
                        help="trained datasets (default: ['C', 'K', 'L', 'N', 'Y', 'Q'])")
    parser.add_argument('--tested_datasets', nargs='+', type=str, default=['C'],
                        help="tested datasets (default: ['C', 'K', 'L', 'N', 'Y', 'Q'])")
    parser.add_argument('--train_proportion', type=float, default=1.0,
                        help='the proportion (#total 100%) used in the training set (default: 1.0)')
    parser.add_argument('--start_exp_id', default=0, type=int,
                        help='strat exp id for train-val-test splits (default: 0)')
    parser.add_argument('--num_iters', type=int, default=10,
                        help='the number of train-val-test iterations (default: 10)')
    args = parser.parse_args()

    for analysis_dataset in args.tested_datasets:
        all_results = []
        for i in range(args.start_exp_id, args.num_iters):
            save_result_file = 'results/{}-{}-{}-{}-{}-{}-{}-{}-{}-EXP{}.npy'.format(args.model, args.feature_extractor, args.loss,
                               args.train_proportion, args.trained_datasets, args.tested_datasets, args.lr, args.batch_size, args.epochs, i)

            test = np.load(save_result_file, allow_pickle=True)
            result = test.tolist()

            iter_mos = result[analysis_dataset]['sq']
            iter_pscore = result[analysis_dataset]['mq']
            # logistic regression
            beta = [np.max(iter_mos), np.min(iter_mos), np.mean(iter_pscore), 0.5]
            popt, _ = curve_fit(logistic_func, iter_pscore, iter_mos, p0=beta, maxfev=100000000)
            iter_pscore_logistic = logistic_func(iter_pscore, *popt)
            SRCC = stats.spearmanr(iter_mos, iter_pscore)[0]
            PLCC = stats.pearsonr(iter_mos, iter_pscore_logistic)[0]

            result[analysis_dataset]['SROCC'] = SRCC
            result[analysis_dataset]['PLCC'] = PLCC
            all_results.append(result[analysis_dataset])

        srcc = []
        plcc = []
        for i in range(args.start_exp_id, args.num_iters):
            srcc.append(all_results[i]['SROCC'])
            plcc.append(all_results[i]['PLCC'])
        print('Performance on {}:'.format(analysis_dataset))
        print('median_srcc: {}, median_plcc: {}'.format(np.median(srcc), np.median(plcc)))
        print('mean_srcc: {}, mean_plcc: {}'.format(np.mean(srcc), np.mean(plcc)))

