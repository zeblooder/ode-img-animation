import os
import importlib
import sys
from scipy.spatial import ConvexHull
from sync_batchnorm import DataParallelWithCallback
import torch
from skimage import img_as_ubyte
from skimage.transform import resize
import numpy as np
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
import matplotlib

matplotlib.use('Agg')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoint(config, checkpoint_path, algo, cpu=False):
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
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))
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


def load_checkpoints(config, methods, cpu=False):
    if len(methods) == 1:
        return load_checkpoint(config, config['pretrained_paths'][methods[0]], methods[0], cpu)
    else:
        generator_dict = {}
        kp_detector_dict = {}
        for m in methods:
            generator_dict[m], kp_detector_dict[m] = load_checkpoint(config, config['pretrained_paths'][m], m, cpu)
        return generator_dict, kp_detector_dict


def generate(generator, kp_detector, source, driving_frame):
    source_tensor = torch.tensor(source[np.newaxis].astype(np.float32)).cuda()
    with torch.no_grad():
        kp_source = kp_detector(source_tensor)
        kp_driving = kp_detector(torch.tensor(driving_frame[np.newaxis].astype(np.float32)).cuda())
        out = generator(source_tensor, kp_source=kp_source, kp_driving=kp_driving)
    return out['prediction'].data.cpu().numpy()[0]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--source_image_dir",
                        default='source_image.txt', help="path to source image")
    parser.add_argument(
        "--driving_img_dir", default='driving_video.txt', help="path to driving video")
    parser.add_argument("--result_image", default='', help="path to output")
    parser.add_argument("--dir", default='res', help="dir to output")

    parser.add_argument("--methods", default=[], type=lambda x: eval(x),
                        help="Names of the methods comma separated.")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu",
                        action="store_true", help="cpu mode.")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    if not os.path.exists(opt.dir):
        os.makedirs(opt.dir)
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    generator_dict, kp_detector_dict = load_checkpoints(config, opt.methods)

    source_image_list = [os.path.join(opt.source_image_dir, p) for p in os.listdir(opt.source_image_dir)]
    driving_image_list = [os.path.join(opt.driving_img_dir, p) for p in os.listdir(opt.driving_img_dir)]

    for source_image_name, driving_image_name in zip(source_image_list, driving_image_list):
        source_image = imageio.imread(source_image_name)
        driving_image = imageio.imread(driving_image_name)

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_image = resize(driving_image, (256, 256))[..., :3]
        if len(opt.result_image) == 0:
            name1 = os.path.split(source_image_name)[1][:-4]
            name2 = os.path.split(driving_image_name)[1][:-4]
            result_image = [os.path.join(
                opt.dir, name1 + '_' + name2 + '_' + m + '.png') for m in opt.methods]
        else:
            result_image = [os.path.join(
                opt.dir, opt.result_image) for m in opt.methods]

        predictions = generate(generator_dict, kp_detector_dict, np.transpose(source_image,[2,0,1]), np.transpose(driving_image,[2,0,1]))
        imageio.imsave(result_image[0],(255 * predictions).transpose(1, 2, 0).astype(np.uint8))
