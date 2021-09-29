"""Extracting Video Motion Features using model-based transfer learning"""

from argparse import ArgumentParser
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
import h5py
import numpy as np
import random
import time
import os

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height, self.width, inputdict={'-pix_fmt':'yuvj420p'})
        else:
            video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        video_score = self.score[idx]

        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        print('video_width: {} video_height: {}'.format(video_width, video_height))

        sample = {'video': video_data, 'score': video_score}

        return sample

class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self, model='ResNet-50'):
        super(CNNModel, self).__init__()
        if model == 'MotionExtractor':
            print("use MotionExtractor")
            from MotionExtractor.get_motionextractor_model import make_motion_model
            model = make_motion_model()
            self.features = model
            self.model = 'MotionExtractor'
        else:
            print("use default ResNet-50")
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
            self.model = 'ResNet-50'

    def forward(self, x):
        x = self.features(x)

        if self.model == 'MotionExtractor':
            features_mean = nn.functional.adaptive_avg_pool2d(x[1], 1)
            features_std = global_std_pool3d(x[1])
            features_mean = torch.squeeze(features_mean).permute(1, 0)
            features_std = torch.squeeze(features_std).permute(1, 0)
        else:
            features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
            features_std = global_std_pool2d(x)

        return features_mean, features_std

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)

def global_std_pool3d(x):
    """3D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], x.size()[2], -1, 1), dim=3, keepdim=True)

from MotionExtractor.slowfast.visualization.utils import process_cv2_inputs
from MotionExtractor.slowfast.utils.parser import load_config, parse_args
def get_features(video_data, frame_batch_size=64, model='ResNet-50', device='cuda'):
    """feature extraction"""
    extractor = CNNModel(model=model).to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        if model == 'MotionExtractor':
            args = parse_args()
            cfg = load_config(args)
            if video_length <= frame_batch_size:
                batch = video_data[0:video_length]
                inputs = process_cv2_inputs(batch, cfg)
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda()

                features_mean, features_std = extractor(inputs)
                output1 = torch.cat((output1, features_mean), 0)
                output2 = torch.cat((output2, features_std), 0)
                output = torch.cat((output1, output2), 1).squeeze()
            else:
                num_block = 0
                while frame_end < video_length:
                    batch = video_data[frame_start:frame_end]
                    inputs = process_cv2_inputs(batch, cfg)
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda()

                    features_mean, features_std = extractor(inputs)
                    output1 = torch.cat((output1, features_mean), 0)
                    output2 = torch.cat((output2, features_std), 0)
                    frame_end += frame_batch_size
                    frame_start += frame_batch_size
                    num_block = num_block + 1

                last_batch = video_data[(video_length - frame_batch_size):video_length]
                inputs = process_cv2_inputs(last_batch, cfg)
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda()

                features_mean, features_std = extractor(inputs)
                index = torch.linspace(0, (frame_batch_size - 1), 32).long()
                last_batch_index = (video_length - frame_batch_size) + index
                elements = torch.where(last_batch_index >= frame_batch_size * num_block)
                output1 = torch.cat((output1, features_mean[elements[0], :]), 0)
                output2 = torch.cat((output2, features_std[elements[0], :]), 0)
                output = torch.cat((output1, output2), 1).squeeze()
        else:
            while frame_end < video_length:
                batch = video_data[frame_start:frame_end].to(device)
                features_mean, features_std = extractor(batch)
                output1 = torch.cat((output1, features_mean), 0)
                output2 = torch.cat((output2, features_std), 0)
                frame_end += frame_batch_size
                frame_start += frame_batch_size

            last_batch = video_data[frame_start:video_length].to(device)
            features_mean, features_std = extractor(last_batch)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            output = torch.cat((output1, output2), 1).squeeze()

    if output.ndim == 1:
        output = output.unsqueeze(0)

    return output

if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting Video Motion Features using model-based transfer learning')
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument('--database', default='CVD2014', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='MotionExtractor', type=str,
                        help='which pre-trained model used (default: ResNet-50)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument("--ith", type=int, default=0, help='start video id')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = 'KoNViD-1k/'  # videos dir, e.g., ln -s /xxx/KoNViD-1k/ KoNViD-1k
        features_dir = 'CNN_features_KoNViD-1k/MotionFeature/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/'
        features_dir = 'CNN_features_CVD2014/MotionFeature/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/'
        features_dir = 'CNN_features_LIVE-Qualcomm/MotionFeature/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/'
        features_dir = 'CNN_features_LIVE-VQC/MotionFeature/'
        datainfo = 'data/LIVE-VQCinfo.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)

    max_len = 0
    min_len = 100000
    for i in range(args.ith, len(dataset)):
        start = time.time()
        current_data = dataset[i]
        print('Video {}: length {}'.format(i, current_data['video'].shape[0]))
        if max_len < current_data['video'].shape[0]:
            max_len = current_data['video'].shape[0]
        if min_len > current_data['video'].shape[0]:
            min_len = current_data['video'].shape[0]
        features = get_features(current_data['video'], args.frame_batch_size, args.model, device)
        np.save(features_dir + str(i) + '_' + args.model + '_last_conv', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_data['score'])
        end = time.time()
        print('{} seconds'.format(end - start))
    print('Max length: {} Min length: {}'.format(max_len, min_len))