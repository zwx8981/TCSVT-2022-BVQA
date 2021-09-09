import torch
from SpatialExtractor.BaseCNN_4FeatureGetting import BaseCNN
from SpatialExtractor.Main_4FeatureGetting import parse_config

def make_spatial_model():

    config = parse_config()
    model = BaseCNN(config)
    model = torch.nn.DataParallel(model).cuda()
    ckpt = './SpatialExtractor/weights/DataParallel-00008.pt'
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])

    return model

if __name__ == "__main__":
    make_spatial_model()