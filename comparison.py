import importlib
import os
import random
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
from tqdm import tqdm
import yaml

from frames_dataset import FramesDataset
from sync_batchnorm import DataParallelWithCallback


def load_checkpoint(config_path, checkpoint_path, algo, cpu=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    Algo = importlib.import_module("packages." + algo)
    generator = Algo.generator.OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                       **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = Algo.keypoint_detector.KPDetector(**config['model_params']['kp_detector_params'],
                                                    **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def load_checkpoints(config_path, methods, cpu=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    generator_dict = {}
    kp_detector_dict = {}
    for m in methods:
        generator_dict[m], kp_detector_dict[m] = load_checkpoint(config_path, config['pretrained_paths'][m], m, cpu)
    return generator_dict, kp_detector_dict


def generate(generator, kp_detector, source, driving_frame):
    source_tensor = torch.tensor(source[np.newaxis].astype(np.float32)).cuda()
    with torch.no_grad():
        kp_source = kp_detector(source_tensor)
        kp_driving = kp_detector(torch.tensor(driving_frame[np.newaxis].astype(np.float32)).cuda())
        out = generator(source_tensor, kp_source=kp_source, kp_driving=kp_driving)
    return out['prediction'].data.cpu().numpy()[0]


def visualize_comparison(generator_dict, kp_detector_dict, methods, source_image, dest_image):
    res = [source_image.transpose([1, 2, 0]) * 255.0, dest_image.transpose([1, 2, 0]) * 255.0]
    for m in methods:
        res.append(
            generate(generator_dict[m], kp_detector_dict[m], source_image, dest_image).transpose([1, 2, 0]) * 255.0)
    image = np.hstack(res).astype(np.uint8)
    return image


def random_img_pair(dataset):
    rand_list = random.sample(range(len(dataset)), 2)
    src_video = dataset.__getitem__(rand_list[0])['video']
    dest_video = dataset.__getitem__(rand_list[1])['video']
    src_frame_index = np.random.randint(src_video.shape[1])
    src_image = src_video[:, src_frame_index, :, :]
    dest_frame_index = np.random.randint(dest_video.shape[1])
    dest_image = dest_video[:, dest_frame_index, :, :]
    name1 = os.path.splitext(dataset.videos[rand_list[0]])[0]
    name2 = os.path.splitext(dataset.videos[rand_list[1]])[0]
    return src_image, dest_image, name1 + '_' + str(src_frame_index) + '_' + name2 + '_' + str(
        dest_frame_index) + '_' + '.png'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--methods", default=[], type=lambda x: eval(x),
                        help="Names of the methods comma separated.")
    parser.add_argument("--path", default=None, help="path to result image")
    parser.add_argument("--num", default=1, type=int, help="number of image")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    dataset = FramesDataset(is_train=False, **config['dataset_params'])
    generator_dict, kp_detector_dict = load_checkpoints(opt.config, opt.methods)
    for i in tqdm(range(opt.num)):
        source_image, dest_image, default_fname = random_img_pair(dataset)
        image = visualize_comparison(generator_dict, kp_detector_dict, opt.methods, source_image, dest_image)
        imageio.imsave(default_fname if opt.path is None else opt.path, image)
