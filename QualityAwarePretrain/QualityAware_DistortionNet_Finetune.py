import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from scipy import stats
import random
import numpy as np
import time
import scipy.io
from torch.optim import lr_scheduler
import torch.nn.functional as F

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DBCNN(torch.nn.Module):

    def __init__(self, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        self.features1 = torchvision.models.resnet50(pretrained=True)

        # Global pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # Linear classifier.
        self.fc_D = torch.nn.Linear(2048, 25)

        # Linear classifier.
        self.fc_DL = torch.nn.Linear(2048, 1)

        # Linear regression.
        self.fc_Q = torch.nn.Linear(2048, 1)

        if options['fc'] == True:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            # Freeze all studied FC layers.
            for param in self.fc_D.parameters():
                param.requires_grad = False
            for param in self.fc_DL.parameters():
                param.requires_grad = False
            # Initial
            nn.init.kaiming_normal_(self.fc_Q.weight.data)
            if self.fc_Q.bias is not None:
                nn.init.constant_(self.fc_Q.bias.data, val=0)

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]

        X1 = self.features1.conv1(X)
        X1 = self.features1.bn1(X1)
        X1 = self.features1.relu(X1)
        X1 = self.features1.maxpool(X1)
        X1 = self.features1.layer1(X1)
        X1 = self.features1.layer2(X1)
        X1 = self.features1.layer3(X1)
        X1 = self.features1.layer4(X1)

        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 2048

        X1 = self.pooling(X1)
        assert X1.size() == (N, 2048, 1, 1)

        X1 = X1.view(N, 2048)
        predict_D = self.fc_D(X1)
        predict_DL = self.fc_DL(X1)
        predict_Q = self.fc_Q(X1)
        assert predict_D.size() == (N, 25)
        assert predict_DL.size() == (N, 1)
        assert predict_Q.size() == (N, 1)

        return predict_D, predict_DL, predict_Q

class DBCNNManager(object):
    def __init__(self, options, path):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """
        print('Prepare the network and data.')
        self._options = options
        self._path = path

        # Network.
        self._net = torch.nn.DataParallel(DBCNN(self._options), device_ids=[0]).cuda()
        if self._options['fc'] == True:
            self._net.load_state_dict(torch.load(path['pretrainkadis700k_root']))
        if self._options['fc'] == False:
            self._net.load_state_dict(torch.load(path['fc_root']))

        print(self._net)
        # Criterion.
        self._criterion_D = torch.nn.CrossEntropyLoss().cuda()
        self._criterion_DL = self.loss_m
        self._criterion_Q = torch.nn.MSELoss().cuda()
        # Solver.
        if self._options['fc'] == True:
            self._solver = torch.optim.SGD(
                self._net.module.fc_Q.parameters(), lr=self._options['base_lr'],
                momentum=0.9, weight_decay=self._options['weight_decay'])
        else:
            self._solver = torch.optim.Adam(
                self._net.module.parameters(), lr=self._options['base_lr'],
                weight_decay=self._options['weight_decay'])

        if self._options['dataset'] == 'kadid10k':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        if self._options['dataset'] == 'kadid10k':
            import Kadid10kFolder_DistortionNet_Finetune
            train_data = Kadid10kFolder_DistortionNet_Finetune.Kadid10kFolder_DistortionNet_Finetune(
                root=self._path['kadid10k'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = Kadid10kFolder_DistortionNet_Finetune.Kadid10kFolder_DistortionNet_Finetune(
                root=self._path['kadid10k'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        else:
            raise AttributeError('Only support KADID-10k right now!')
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)
        self.scheduler = lr_scheduler.StepLR(self._solver,
                                             last_epoch=-1,
                                             step_size=10,
                                             gamma=0.1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        """Train the network."""
        print('Training.')
        best_srcc = 0.0
        best_plcc = 0.0
        best_acc = 0.0
        print('Epoch\tLr\tTotal loss\tCross loss\tHinge loss\tMSE loss\tTrain_ACC\tTest_ACC\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self._options['epochs']):
            self._net.train(True)  # Set the model to training phase
            time_start = time.time()

            epoch_loss = []
            pscores = []
            tscores = []
            num_total = 0.0
            num_correct = 0.0
            cross_loss = []
            hinge_loss = []
            mse_loss = []
            for sample in self._train_loader:
                # Data.
                I1, I2, I3, I4, I5, I1_D, I2_D, I3_D, I4_D, I5_D, I1_DL, I2_DL, I3_DL, I4_DL, I5_DL, I1_M, I2_M, I3_M, I4_M, I5_M = \
                    sample['I1'], sample['I2'], sample['I3'], sample['I4'], sample['I5'],\
                    sample['I1_D'], sample['I2_D'], sample['I3_D'], sample['I4_D'], sample['I5_D'],\
                    sample['I1_DL'], sample['I2_DL'], sample['I3_DL'], sample['I4_DL'], sample['I5_DL'],\
                    sample['I1_M'], sample['I2_M'], sample['I3_M'], sample['I4_M'], sample['I5_M']

                I1 = I1.to(self.device)
                I2 = I2.to(self.device)
                I3 = I3.to(self.device)
                I4 = I4.to(self.device)
                I5 = I5.to(self.device)
                I1_D = I1_D.to(self.device)
                I2_D = I2_D.to(self.device)
                I3_D = I3_D.to(self.device)
                I4_D = I4_D.to(self.device)
                I5_D = I5_D.to(self.device)
                I1_DL = I1_DL.to(self.device)
                I2_DL = I2_DL.to(self.device)
                I3_DL = I3_DL.to(self.device)
                I4_DL = I4_DL.to(self.device)
                I5_DL = I5_DL.to(self.device)
                I1_M = I1_M.to(self.device)
                I2_M = I2_M.to(self.device)
                I3_M = I3_M.to(self.device)
                I4_M = I4_M.to(self.device)
                I5_M = I5_M.to(self.device)

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                I = torch.Tensor().to(self.device)
                y_D = torch.Tensor().to(self.device).long()
                y_DL = torch.Tensor().to(self.device)
                y_M = torch.Tensor().to(self.device)
                for i in range(0, len(I1)):
                    I = torch.cat((I, I1[i].unsqueeze(0), I2[i].unsqueeze(0), I3[i].unsqueeze(0),
                                   I4[i].unsqueeze(0), I5[i].unsqueeze(0)))
                    y_D = torch.cat((y_D, I1_D[i].unsqueeze(0), I2_D[i].unsqueeze(0), I3_D[i].unsqueeze(0),
                                 I4_D[i].unsqueeze(0), I5_D[i].unsqueeze(0)))
                    y_DL = torch.cat((y_DL, I1_DL[i].unsqueeze(0), I2_DL[i].unsqueeze(0), I3_DL[i].unsqueeze(0),
                                      I4_DL[i].unsqueeze(0), I5_DL[i].unsqueeze(0)))
                    y_M = torch.cat((y_M, I1_M[i].unsqueeze(0), I2_M[i].unsqueeze(0), I3_M[i].unsqueeze(0),
                                     I4_M[i].unsqueeze(0), I5_M[i].unsqueeze(0)))

                predict_D, predict_DL, predict_Q = self._net(I)
                pscores = pscores + predict_Q.cpu().tolist()
                tscores = tscores + y_M.cpu().tolist()
                loss_D = self._criterion_D(predict_D, y_D.detach())
                loss_DL = self._criterion_DL(predict_DL, y_DL.unsqueeze(1).detach())
                loss_Q = self._criterion_Q(predict_Q, y_M.unsqueeze(1).detach())
                loss = loss_D + 0.1*loss_DL + loss_Q
                epoch_loss.append(loss.item())
                cross_loss.append(loss_D.item())
                hinge_loss.append(loss_DL.item())
                mse_loss.append(loss_Q.item())

                _, prediction = torch.max(F.softmax(predict_D.data, dim=1), 1)
                num_total += y_D.size(0)
                num_correct += torch.sum(prediction == y_D)
                # Backward pass.
                loss.backward()
                self._solver.step()

            self._net.eval()
            train_acc = 100 * num_correct.float() / num_total
            test_acc = self._accuracy(self._test_loader)
            train_srcc, _ = stats.spearmanr(pscores, tscores)
            test_srcc, test_plcc = self._consitency(self._test_loader)

            time_end = time.time()
            print('%d epoch done; total time = %f sec' % ((t + 1), (time_end - time_start)))

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                print('*', end='')
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd, 'fc_models', ('net_params' + '_best_srcc' + '.pkl'))
                else:
                    modelpath = os.path.join(pwd, 'db_models', ('net_params' + '_best_srcc' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)
            if test_acc > best_acc:
                best_acc = test_acc
                print('*', end='')
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd, 'fc_models', ('net_params' + '_best_acc' + '.pkl'))
                else:
                    modelpath = os.path.join(pwd, 'db_models', ('net_params' + '_best_acc' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)

            if (t+1) == self._options['epochs']:
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd, 'fc_models', ('net_params' + '_latest' + '.pkl'))
                else:
                    modelpath = os.path.join(pwd, 'db_models', ('net_params' + '_latest' + '.pkl'))
                torch.save(self._net.state_dict(), modelpath)

            print('%d\t\t%4.10f\t\t%4.3f\t\t%4.3f\t\t%4.3f\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, self._solver.param_groups[0]['lr'], sum(epoch_loss) / len(epoch_loss), sum(cross_loss) / len(cross_loss),
                   sum(hinge_loss) / len(hinge_loss), sum(mse_loss) / len(mse_loss), train_acc, test_acc, train_srcc, test_srcc, test_plcc))

            self.scheduler.step()
            # if self._options['fc'] != True:
            #     self.scheduler.step()

        return best_srcc, best_plcc

    def _consitency(self, data_loader):
        pscores = []
        tscores = []
        for sample in data_loader:
            # Data.
            I1, I2, I3, I4, I5, I1_M, I2_M, I3_M, I4_M, I5_M = \
                sample['I1'], sample['I2'], sample['I3'], sample['I4'], sample['I5'], \
                sample['I1_M'], sample['I2_M'], sample['I3_M'], sample['I4_M'], sample['I5_M']

            I1 = I1.to(self.device)
            I2 = I2.to(self.device)
            I3 = I3.to(self.device)
            I4 = I4.to(self.device)
            I5 = I5.to(self.device)
            I1_M = I1_M.to(self.device)
            I2_M = I2_M.to(self.device)
            I3_M = I3_M.to(self.device)
            I4_M = I4_M.to(self.device)
            I5_M = I5_M.to(self.device)

            # Prediction.
            I = torch.Tensor().to(self.device)
            y_M = torch.Tensor().to(self.device)
            for i in range(0, len(I1)):
                I = torch.cat((I, I1[i].unsqueeze(0), I2[i].unsqueeze(0), I3[i].unsqueeze(0),
                               I4[i].unsqueeze(0), I5[i].unsqueeze(0)))
                y_M = torch.cat((y_M, I1_M[i].unsqueeze(0), I2_M[i].unsqueeze(0), I3_M[i].unsqueeze(0),
                                 I4_M[i].unsqueeze(0), I5_M[i].unsqueeze(0)))

            predict_D, predict_DL, predict_Q = self._net(I)
            pscores = pscores + predict_Q[:, 0].cpu().tolist()
            tscores = tscores + y_M.cpu().tolist()

        test_srcc, _ = stats.spearmanr(pscores, tscores)
        tscores = torch.Tensor(tscores).reshape(-1).tolist()
        test_plcc, _ = stats.pearsonr(pscores, tscores)

        return test_srcc, test_plcc

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.
        Args:
            data_loader: Train/Test DataLoader.
        Returns:
            Train/Test accuracy in percentage.
        """
        num_correct = 0.0
        num_total = 0.0
        for sample in data_loader:
            # Data.
            I1, I2, I3, I4, I5, I1_D, I2_D, I3_D, I4_D, I5_D = \
                  sample['I1'], sample['I2'], sample['I3'], sample['I4'], sample['I5'], \
                  sample['I1_D'], sample['I2_D'], sample['I3_D'], sample['I4_D'], sample['I5_D']

            I1 = I1.to(self.device)
            I2 = I2.to(self.device)
            I3 = I3.to(self.device)
            I4 = I4.to(self.device)
            I5 = I5.to(self.device)
            I1_D = I1_D.to(self.device)
            I2_D = I2_D.to(self.device)
            I3_D = I3_D.to(self.device)
            I4_D = I4_D.to(self.device)
            I5_D = I5_D.to(self.device)

            # Prediction.
            I = torch.Tensor().to(self.device)
            y_D = torch.Tensor().to(self.device).long()
            for i in range(0, len(I1)):
                I = torch.cat((I, I1[i].unsqueeze(0), I2[i].unsqueeze(0), I3[i].unsqueeze(0),
                               I4[i].unsqueeze(0), I5[i].unsqueeze(0)))
                y_D = torch.cat((y_D, I1_D[i].unsqueeze(0), I2_D[i].unsqueeze(0), I3_D[i].unsqueeze(0),
                                 I4_D[i].unsqueeze(0), I5_D[i].unsqueeze(0)))

            predict_D, predict_DL, predict_Q = self._net(I)

            _, prediction = torch.max(predict_D.data, 1)
            num_total += y_D.size(0)
            num_correct += torch.sum(prediction == y_D.data)

        return 100 * num_correct.float() / num_total

    def loss_m(self, y_pred, y):
        """prediction monotonicity related loss"""
        assert y_pred.size(0) > 1
        loss = torch.Tensor().to(self.device)
        #for i in range(0, self._options['batch_size']):
        for i in range(0, (y_pred.size(0) // 5)):
            y_pred_one = y_pred[i*5:(i+1)*5, :]
            y_one = y[i*5:(i+1)*5, :]
            tmp = F.relu((y_pred_one - (y_pred_one+10).t()) * torch.sign((y_one.t() - y_one)))
            loss = torch.cat((loss, tmp.unsqueeze(0)), 0)

        return torch.mean(loss.view(-1, 1))

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train DB-CNN for BIQA.')
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-6,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=8, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=20, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadid10k',
                        help='dataset: kadid10k')

    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset': args.dataset,
        'fc': [],
        'train_index': [],
        'test_index': []
    }

    path = {
        'kadid10k': os.path.join('dataset', '/mnt/sda2/New/kadid10k'),
        'pretrainkadis700k_root': os.path.join('db_models/kadis700k', 'net_params_best_srcc.pkl'),
        'fc_model': os.path.join('fc_models'),
        'fc_root': os.path.join('fc_models', 'net_params_latest.pkl'),
        'db_model': os.path.join('db_models'),
        'db_root': os.path.join('db_models', 'net_params_latest.pkl')
    }

    if options['dataset'] == 'kadid10k':
        index = list(range(0, 81))  # 81

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    lr_backup = options['base_lr']
    iter_num = 1
    srcc_all = np.zeros((1, iter_num + 1), dtype=np.float)
    plcc_all = np.zeros((1, iter_num + 1), dtype=np.float)

    for i in range(0, iter_num):
        # randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8 * len(index))]
        test_index = index[round(0.8 * len(index)):len(index)]

        d_type_num = 23
        train_index_new = []
        for train_index_idx in train_index:
            train_index_new = train_index_new + np.arange(train_index_idx * d_type_num, (train_index_idx+1) * d_type_num).tolist()
        test_index_new = []
        for test_index_idx in test_index:
            test_index_new = test_index_new + np.arange(test_index_idx * d_type_num, (test_index_idx + 1) * d_type_num).tolist()

        train_index = train_index_new
        test_index = test_index_new
        options['train_index'] = train_index
        options['test_index'] = test_index
        pwd = os.getcwd()
        # train FC layers only
        options['fc'] = True
        options['base_lr'] = 1e-3
        options['epochs'] = 20
        manager = DBCNNManager(options, path)
        best_srcc, best_plcc = manager.train()

        # fine-tune all model
        options['fc'] = False
        options['base_lr'] = lr_backup
        options['epochs'] = 20
        manager = DBCNNManager(options, path)
        best_srcc, best_plcc = manager.train()
        result_path = os.path.join(pwd, 'result', ('db_result_' + str(i + 1) + '.mat'))
        scipy.io.savemat(result_path, mdict={'best_srcc': best_srcc, 'best_plcc': best_plcc})

        srcc_all[0][i] = best_srcc
        plcc_all[0][i] = best_plcc

    srcc_mean = np.mean(srcc_all[0][0:iter_num])
    plcc_mean = np.mean(plcc_all[0][0:iter_num])
    print('\n Average srcc:%4.4f, Average plcc:%4.4f' % (srcc_mean, plcc_mean))
    srcc_all[0][iter_num] = srcc_mean
    plcc_all[0][iter_num] = plcc_mean
    print(srcc_all)
    print(plcc_all)

    final_result_path = os.path.join(pwd, 'result', ('final_result' + '.mat'))
    scipy.io.savemat(final_result_path, mdict={'srcc_all': srcc_all, 'plcc_all': plcc_all})

    return best_srcc


if __name__ == '__main__':
    main()
