import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import importlib

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from frames_dataset import FramesDataset

import torch

from train import train
from reconstruction import reconstruction
from animate import animate
from performance import performance

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate", "evaluation"])
    parser.add_argument("--algo", default="gaussian", help="algorithm")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--metrics", default="0", type=lambda x: list(map(str, x.split(','))),
                        help="Names of the metrics comma separated.")
    parser.add_argument("--result_table", default=None, help="path to output")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()

    Algo = importlib.import_module("packages." + opt.algo)

    with open(opt.config) as f:
        config = yaml.safe_load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = Algo.generator.OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                       **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = Algo.discriminator.MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                               **config['model_params']['common_params'])
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    kp_detector = Algo.keypoint_detector.KPDetector(**config['model_params']['kp_detector_params'],
                                                    **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])

    if opt.verbose:
        print(kp_detector)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)
    result_table = opt.algo + '.csv' if opt.result_table is None else opt.result_table
    print("Model: " + str(Algo) + ".\nResult table name: " + result_table)

    if opt.mode == 'train':
        print("Training...")
        train(opt.algo, config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(opt.algo, config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(opt.algo, config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'evaluation':
        print("Evaluation...")
        performance(generator, kp_detector, config['pretrained_paths'][opt.algo], dataset, opt.metrics, result_table)
