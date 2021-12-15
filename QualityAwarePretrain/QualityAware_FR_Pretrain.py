import os

import torch
import torchvision
import torch.nn as nn
from SCNN import SCNN
from PIL import Image
from scipy import stats
import random
import torch.nn.functional as F
import numpy as np
import time
import scipy.io
import itertools
from torch.optim import lr_scheduler

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

def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class DBCNN(torch.nn.Module):

    def __init__(self, options):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # self.features1 = torchvision.models.resnet34(pretrained=True)
        self.features1 = torchvision.models.resnet50(pretrained=True)
        # weight_init(self.features1)

        # Global pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # Linear classifier.
        self.fc = torch.nn.Linear(2048, 1)

        if options['fc'] == True:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

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
        X = self.fc(X1)
        assert X.size() == (N, 1)

        return X

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
        if self._options['fc'] == False:
            self._net.load_state_dict(torch.load(path['fc_root']))

        print(self._net)
        # Criterion.
        self._criterion = torch.nn.MSELoss().cuda()

        # Solver.
        if self._options['fc'] == True:
            self._solver = torch.optim.SGD(
                self._net.module.fc.parameters(), lr=self._options['base_lr'],
                momentum=0.9, weight_decay=self._options['weight_decay'])
        else:
            self._solver = torch.optim.Adam(
                self._net.module.parameters(), lr=self._options['base_lr'],
                weight_decay=self._options['weight_decay'])

        if (self._options['dataset'] == 'live') | (self._options['dataset'] == 'livec'):
            if self._options['dataset'] == 'live':
                crop_size = 432
            else:
                crop_size = 448
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=crop_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        # if (self._options['dataset'] == 'live'):
        #         if self._options['dataset'] == 'live':
        #             crop_size = 432
        #         else:
        #             crop_size = 448
        #         train_transforms = torchvision.transforms.Compose([
        #             torchvision.transforms.RandomHorizontalFlip(),
        #             torchvision.transforms.RandomCrop(size=crop_size),
        #             torchvision.transforms.ToTensor(),
        #             torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                                              std=(0.229, 0.224, 0.225))
        #         ])
        # elif (self._options['dataset'] == 'livec'):
        #     train_transforms = torchvision.transforms.Compose([
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                                          std=(0.229, 0.224, 0.225))
        #     ])
        elif (self._options['dataset'] == 'csiq') | (self._options['dataset'] == 'tid2013'):
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'mlive':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((570, 960)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'livecp':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'koniq10k':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'kadid10k':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'kadis700k':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
        elif self._options['dataset'] == 'selftrain':
            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=500),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))

                # torchvision.transforms.Resize((384, 512)),
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                #                                  std=(0.229, 0.224, 0.225))
            ])

        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        if self._options['dataset'] == 'live':
            import LIVEFolder
            train_data = LIVEFolder.LIVEFolder(
                root=self._path['live'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = LIVEFolder.LIVEFolder(
                root=self._path['live'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        elif self._options['dataset'] == 'csiq':
            import CSIQFolder
            train_data = CSIQFolder.CSIQFolder(
                root=self._path['csiq'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = CSIQFolder.CSIQFolder(
                root=self._path['csiq'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        # elif self._options['dataset'] == 'livec':
        #     import LIVEChallengeFolder2
        #     train_data = LIVEChallengeFolder2.Koniq10kFolder(
        #         root=self._path['koniq10k'], loader=default_loader, index=self._options['train_index'],
        #         transform=train_transforms)
        #     test_data = LIVEChallengeFolder2.LIVECompressedFolder2(
        #         root=self._path['livec'], loader=default_loader, index=self._options['test_index'],
        #         transform=test_transforms)
        elif self._options['dataset'] == 'livec':
            import LIVEChallengeFolder
            train_data = LIVEChallengeFolder.LIVEChallengeFolder(
                root=self._path['livec'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = LIVEChallengeFolder.LIVEChallengeFolder(
                root=self._path['livec'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        elif self._options['dataset'] == 'livecp':
            import LIVECompressedFolder
            train_data = LIVECompressedFolder.LIVECompressedFolder(
                root=self._path['livecp'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = LIVECompressedFolder.LIVECompressedFolder(
                root=self._path['livecp'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        elif self._options['dataset'] == 'koniq10k':
            import Koniq10kFolder
            train_data = Koniq10kFolder.Koniq10kFolder(
                root=self._path['koniq10k'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = Koniq10kFolder.Koniq10kFolder(
                root=self._path['koniq10k'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        elif self._options['dataset'] == 'kadid10k':
            import Kadid10kFolder
            train_data = Kadid10kFolder.Kadid10kFolder(
                root=self._path['kadid10k'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = Kadid10kFolder.Kadid10kFolder(
                root=self._path['kadid10k'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        elif self._options['dataset'] == 'kadis700k':
            import Kadis700kFolder
            train_data = Kadis700kFolder.Kadis700kFolder(
                root=self._path['kadis700k'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            test_data = Kadis700kFolder.Kadis700kFolder(
                root=self._path['kadis700k'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
        elif self._options['dataset'] == 'selftrain':
            import SelfTrainFolder
            # train_data = SelfTrainFolder.Kadid10kFolder(
            #         root=self._path['kadid10k'], loader = default_loader, index = self._options['train_index'],
            #         transform=train_transforms)
            train_data = SelfTrainFolder.LIVEChallengeFolder(
                root=self._path['livecp'], loader=default_loader, index=self._options['train_index'],
                transform=train_transforms)
            # root = self._path['koniq10k'], loader = default_loader, index = self._options['train_index2'],
            # transform = train_transforms)
            # test_data = SelfTrainFolder.Kadid10kFolder(
            #         root=self._path['kadid10k'], loader = default_loader, index = self._options['test_index'],
            #         transform=test_transforms)
            test_data = SelfTrainFolder.LIVECompressed2Folder(
                root=self._path['livecp'], loader=default_loader, index=self._options['test_index'],
                transform=test_transforms)
            # test_data2 = SelfTrainFolder.Koniq10kFolder(
            #         root=self._path['koniq10k'], loader = default_loader, index = self._options['test_index2'],
            #         transform=test_transforms)
        else:
            raise AttributeError('Only support LIVE and LIVEC right now!')
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=True)
        self.scheduler = lr_scheduler.StepLR(self._solver,
                                             last_epoch=-1,
                                             step_size=2,
                                             gamma=0.1)
    def train(self, iteration):
        """Train the network."""
        print('Training.')
        best_srcc = 0.0
        best_plcc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        for t in range(self._options['epochs']):
            time_start = time.time()

            epoch_loss = []
            pscores = []
            tscores = []
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = torch.tensor(X.cuda())
                y = torch.tensor(y.cuda())

                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y.view(len(score), 1).detach())
                epoch_loss.append(loss.item())
                # Prediction.
                num_total += y.size(0)
                pscores = pscores + score.cpu().tolist()
                tscores = tscores + y.cpu().tolist()
                # Backward pass.
                loss.backward()
                self._solver.step()

            train_srcc, _ = stats.spearmanr(pscores, tscores)
            test_srcc, test_plcc = self._consitency(self._test_loader)

            time_end = time.time()
            print('%d epoch done; total time = %f sec' % ((t + 1), (time_end - time_start)))

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                best_epoch = t + 1
                print('*', end='')
                pwd = os.getcwd()
                if self._options['fc'] == True:
                    modelpath = os.path.join(pwd, 'fc_models', ('net_params' + '_best' + '.pkl'))
                else:
                    modelpath = os.path.join(pwd, 'db_models', ('net_params' + '_best' + '.pkl'))

                torch.save(self._net.state_dict(), modelpath)

            print('%d\t%4.10f\t%4.3f\t\t%4.4f\t\t%4.4f\t%4.4f' %
                  (t + 1, self._solver.param_groups[0]['lr'], sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            if self._options['fc'] != True:
                self.scheduler.step()

        print('Best at epoch %d, test srcc %f' % (best_epoch, best_srcc))

        return best_srcc, best_plcc

    def _consitency(self, data_loader):
        self._net.train(False)
        num_total = 0
        pscores = []
        tscores = []
        for X, y in data_loader:
            # Data.
            X = torch.tensor(X.cuda())
            y = torch.tensor(y.cuda())

            # Prediction.
            score = self._net(X)
            pscores = pscores + score[0].cpu().tolist()
            tscores = tscores + y.cpu().tolist()

            num_total += y.size(0)
        test_srcc, _ = stats.spearmanr(pscores, tscores)
        tscores = torch.Tensor(tscores).reshape(-1).tolist()  # live compressed
        test_plcc, _ = stats.pearsonr(pscores, tscores)
        self._net.train(True)  # Set the model to training phase
        return test_srcc, test_plcc

def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train DB-CNN for BIQA.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-5,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=32, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=30, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay.')
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadis700k',
                        help='dataset: live|csiq|tid2013|livec|mlive|livecp|koniq10k|kadid10k|selftrain|kadis700k')

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
        'live': os.path.join('dataset', 'F:\dataset\databaserelease2'),
        'csiq': os.path.join('dataset', 'S:\dataset\CSIQ'),
        'tid2013': os.path.join('dataset', 'TID2013'),
        # 'livec': os.path.join('dataset', 'F:\\dataset\\ChallengeDB_release\\Images'),
        'livec': os.path.join('dataset', 'F:\\dataset\\ChallengeDB_release'),
        'mlive': os.path.join('dataset', 'LIVEmultidistortiondatabase'),
        'livecp': os.path.join('dataset', 'F:\\dataset\\LIVECompressed2'),  # F:\\dataset\\LIVECompressed2\\level63
        'koniq10k': os.path.join('dataset', 'F:\\dataset\\koniq_10k\\author\\koniq10k_512x384'), # 'F:\\dataset\\LIVECompressed2\\Koniq10k_JPEG_distorted_images'),
        'kadid10k': os.path.join('dataset', 'F:\\dataset\\KADID-10k\\kadid10k'),
        'selftrain': os.path.join('dataset', 'F:\\dataset\\KADID-10k\\kadid10k'),
        'kadis700k': os.path.join('dataset', '/mnt/sda2/New/kadis700k/kadis700k'),
        'fc_model': os.path.join('fc_models'),
        'scnn_root': os.path.join('pretrained_scnn', 'net_params18.pkl'),
        'fc_root': os.path.join('fc_models', 'net_params_best.pkl'),
        'db_model': os.path.join('db_models'),
        'db_root': os.path.join('db_models', 'net_params_best.pkl')
    }

    if options['dataset'] == 'live':
        index = list(range(0, 29))
    elif options['dataset'] == 'csiq':
        index = list(range(0, 30))
    elif options['dataset'] == 'tid2013':
        index = list(range(0, 25))
    elif options['dataset'] == 'mlive':
        index = list(range(0, 15))
    # elif options['dataset'] == 'livec':
    #     index = list(range(0, 10073))
    #     index_test = list(range(0, 80))
    elif options['dataset'] == 'livec':
        index = list(range(0, 1162))
    elif options['dataset'] == 'livecp':
        index = list(range(0, 80))
    elif options['dataset'] == 'koniq10k':
        index = list(range(0, 10073))
    elif options['dataset'] == 'kadid10k':
        index = list(range(0, 10125))
    elif options['dataset'] == 'kadis700k':
        index = list(range(0, 129109))  # 129109
    elif options['dataset'] == 'selftrain':
        index = list(range(0, 1082))
        # index2 = list(range(0, 10073))
        index_test = list(range(0, 80))

    lr_backup = options['base_lr']
    iter_num = 1
    srcc_all = np.zeros((1, iter_num + 1), dtype=np.float)
    plcc_all = np.zeros((1, iter_num + 1), dtype=np.float)

    for i in range(0, iter_num):
        best_srcc = 0.0
        best_plcc = 0.0
        # randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8 * len(index))]
        test_index = index[round(0.8 * len(index)):len(index)]

        options['train_index'] = train_index
        options['test_index'] = test_index
        pwd = os.getcwd()
        # train the fully connected layer only
        options['fc'] = True
        options['base_lr'] = 1e-3
        manager = DBCNNManager(options, path)
        best_srcc, best_plcc = manager.train(i)

        # fine-tune all model
        options['fc'] = False
        options['base_lr'] = lr_backup
        manager = DBCNNManager(options, path)
        best_srcc, best_plcc = manager.train(i)
        result_path = os.path.join(pwd, 'result', ('db_result_' + str(i + 1) + '.mat'))
        scipy.io.savemat(result_path, mdict={'best_srcc': best_srcc, 'best_plcc': best_plcc})

        srcc_all[0][i] = best_srcc
        plcc_all[0][i] = best_plcc

    srcc_mean = np.mean(srcc_all[0][0:iter_num])
    print('average srcc:%4.4f' % (srcc_mean))
    plcc_mean = np.mean(plcc_all[0][0:iter_num])
    print('average plcc:%4.4f' % (plcc_mean))
    srcc_all[0][iter_num] = srcc_mean
    plcc_all[0][iter_num] = plcc_mean
    print(srcc_all)
    print(plcc_all)

    final_result_path = os.path.join(pwd, 'result', ('final_result' + '.mat'))
    scipy.io.savemat(final_result_path, mdict={'srcc_all': srcc_all, 'plcc_all': plcc_all})

    return best_srcc


if __name__ == '__main__':
    main()
