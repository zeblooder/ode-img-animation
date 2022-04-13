import matplotlib

matplotlib.use('Agg')
import sys, importlib, os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from scipy.spatial import ConvexHull

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


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


def load_checkpoints(config, methods, cpu=False):
    if len(methods) == 1:
        return load_checkpoint(config, config['pretrained_paths'][methods[0]], methods[0], cpu)
    else:
        generator_dict = {}
        kp_detector_dict = {}
        for m in methods:
            generator_dict[m], kp_detector_dict[m] = load_checkpoint(config, config['pretrained_paths'][m], m, cpu)
        return generator_dict, kp_detector_dict

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            # kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
            #                        kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
            #                        use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_driving)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image_file", default='source_image.txt', help="path to source image")
    parser.add_argument("--driving_video_file", default='driving_video.txt', help="path to driving video")
    parser.add_argument("--result_video", default='', help="path to output")
    parser.add_argument("--dir", default='res', help="dir to output")

    parser.add_argument("--methods", default=[], type=lambda x: eval(x),
                        help="Names of the methods comma separated.")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    if not os.path.exists(opt.dir):
        os.makedirs(opt.dir)
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    generator_dict, kp_detector_dict = load_checkpoints(config, opt.methods)

    with open(opt.source_image_file) as f:
        source_image_list=[line.strip() for line in f.readlines()]
    with open(opt.driving_video_file) as f:
        driving_video_list=[line.strip() for line in f.readlines()]

    for source_image_name, driving_video_name in zip(source_image_list,driving_video_list):
        source_image = imageio.imread(source_image_name)
        reader = imageio.get_reader(driving_video_name)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        if len(opt.result_video)==0:
            name1 = os.path.split(source_image_name)[1][:-4]
            name2 = os.path.split(driving_video_name)[1][:-4]
            result_video=[os.path.join(opt.dir,name1 + '_' + name2 + '_' + m + '.gif') for m in opt.methods]
        else:
            result_video=[os.path.join(opt.dir,opt.result_video) for m in opt.methods]

        if opt.find_best_frame or opt.best_frame is not None:
            i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
            print("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            for i, m in enumerate(opt.methods):
                predictions_forward = make_animation(source_image, driving_forward, generator_dict[m], kp_detector_dict[m],
                                                    relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
                predictions_backward = make_animation(source_image, driving_backward, generator_dict[m], kp_detector_dict[m],
                                                    relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
                predictions = predictions_backward[::-1] + predictions_forward[1:]
                imageio.mimsave(result_video[i], [img_as_ubyte(frame) for frame in predictions], fps=fps)
        else:
            for i, m in enumerate(opt.methods):
                predictions = make_animation(source_image, driving_video, generator_dict[m], kp_detector_dict[m], relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
                imageio.mimsave(result_video[i], [img_as_ubyte(frame) for frame in predictions], fps=fps)