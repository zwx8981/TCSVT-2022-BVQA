"""Video Feature Fusion"""

from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import random
import time
import os

def fuse_features(features1, features2, frame_batch_size=64, device='cuda'):
    """feature fusion"""
    video_length = features1.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    features = np.empty(shape=[0, 4096], dtype=np.float32)

    if video_length <= frame_batch_size:
        index = torch.linspace(0, (features1.shape[0] - 1), 32).long()
        index = index.to(device)
        features = torch.index_select(features1, 0, index)
    else:
        index = torch.linspace(0, (frame_batch_size - 1), 32).long().numpy()
        num_block = 0
        while frame_end < video_length:
            batch = features1[frame_start:frame_end, :]
            features = np.concatenate((features, batch[index, :]), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size
            num_block = num_block + 1

        # last_batch_index = ((video_length - frame_batch_size) + index)
        # elements = last_batch_index[(last_batch_index >= frame_batch_size * num_block)]
        # features = np.concatenate((features, features1[elements, :]), 0)

        last_batch_index = (video_length - frame_batch_size) + index
        elements = np.where(last_batch_index >= frame_batch_size * num_block)
        elements = elements[0] + (video_length - frame_batch_size)
        features = np.concatenate((features, features1[elements, :]), 0)

    features = np.concatenate((features, features2), 1)

    return features

if __name__ == "__main__":
    parser = ArgumentParser(description='Video Feature Fusion')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='CVD2014', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='SpatialMotion', type=str,
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
        features_dir = 'CNN_features_KoNViD-1k/'  # features dir
        datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'CVD2014':
        videos_dir = 'CVD2014/'
        features_dir = 'CNN_features_CVD2014/'
        datainfo = 'data/CVD2014info.mat'
    if args.database == 'LIVE-Qualcomm':
        videos_dir = 'LIVE-Qualcomm/'
        features_dir = 'CNN_features_LIVE-Qualcomm/'
        datainfo = 'data/LIVE-Qualcomminfo.mat'
    if args.database == 'LIVE-VQC':
        videos_dir = 'LIVE-VQC/'
        features_dir = 'CNN_features_LIVE-VQC/'
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

    for i in range(args.ith, len(video_names)):
        start = time.time()
        print('Video: {}'.format(i))

        spatial_features = np.load(features_dir + 'SpatialFeature/' + str(i) + '_' + 'SpatialExtractor' + '_last_conv.npy')
        motion_features = np.load(features_dir + 'MotionFeature/' + str(i) + '_' + 'MotionExtractor' + '_last_conv.npy')
        score = np.load(features_dir + 'SpatialFeature/' + str(i) + '_score.npy')
        features = fuse_features(spatial_features, motion_features, args.frame_batch_size)
        np.save(features_dir + str(i) + '_' + args.model +'_last_conv', features)
        np.save(features_dir + str(i) + '_score', score)

        end = time.time()
        print('{} seconds'.format(end-start))
