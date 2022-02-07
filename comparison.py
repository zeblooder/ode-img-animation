import importlib
import os
import random
from argparse import ArgumentParser

import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

from frames_dataset import FramesDataset
from sync_batchnorm import DataParallelWithCallback


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


def random_img_pair(dataset,mode,seed=0):
    random.seed(seed)
    rand_list = random.sample(range(len(dataset)), 2)
    src_video = dataset.__getitem__(rand_list[0])['video']
    if mode=="rec":
        dest_video =src_video
    elif mode=="anim":
        dest_video = dataset.__getitem__(rand_list[1])['video']
    src_frame_index = np.random.randint(src_video.shape[1])
    src_image = src_video[:, src_frame_index, :, :]
    dest_frame_index = np.random.randint(dest_video.shape[1])
    dest_image = dest_video[:, dest_frame_index, :, :]
    name1 = os.path.splitext(dataset.videos[rand_list[0]])[0]
    name2 = os.path.splitext(dataset.videos[rand_list[1]])[0]
    return src_image, dest_image, name1 + '_' + str(src_frame_index) + '_' + name2 + '_' + str(
        dest_frame_index) + '_' + '.png'


def random_specified_num_img_pair(dataset, col, row, seed=0, source_index=0):
    """
    generate images for showing results in a table
    """
    random.seed(seed)
    rand_list = random.sample(range(len(dataset)), row)
    driving_video = dataset.__getitem__(rand_list[0])['video']
    source_videos = [dataset.__getitem__(rand_list[i])['video'] for i in range(row)]
    dest_frame_index = [i + 1 for i in sorted(random.sample(range(driving_video.shape[1] - 1), col))]
    source_images = [source_videos[i][:, source_index, :, :] for i in range(row)]
    driving_images = [driving_video[:, i, :, :] for i in dest_frame_index]
    return source_images, driving_images


def gen_tab_latex(col, row, method):
    img = "\\includegraphics[width=20mm]{{image/chap04/experiment_src_driv/{}-{}.png}}"
    ret = '\\begin{tabular}{ccccc}\n\\hline\n\\diagbox{Source}{Driving} &\n'
    for i in range(col + 1):
        if i != 0:
            ret += img.format(method, str(i) + '-0') + ' &\n'
        for j in range(row):
            ret += img.format(method, str(i) + '-' + str(j + 1))
            if j + 1 != row:
                ret += ' &\n'
            else:
                ret += ' \\\\\n\\hline\n'
    return ret + '\\end{tabular}'


def gen_table(config, dataset, col=4, row=4, method='gaussian', seed=0):
    source_images, driving_images = random_specified_num_img_pair(dataset, col, row, seed)
    generator, kp_detector = load_checkpoints(config, [method])
    for j in range(col):
        imageio.imsave("{}-0-{}.png".format(method, j + 1),
                       (255 * driving_images[j].transpose(1, 2, 0)).astype(np.uint8))
    for i in range(row):
        imageio.imsave("{}-{}-0.png".format(method, i + 1),
                       (255 * source_images[i].transpose(1, 2, 0)).astype(np.uint8))
        for j in range(col):
            imageio.imsave("{}-{}-{}.png".format(method, i + 1, j + 1),
                           (255 * generate(generator, kp_detector, source_images[i], driving_images[j])).transpose(1, 2, 0).astype(np.uint8))
    return gen_tab_latex(col, row, method)


def gen_compare2(config,dataset,methods,num,path,mode='rec',seed=0,caption='11',label='22'):
    latex_str=r'\begin{figure}{0.55\textwidth}'+'\n'
    latex_str+="    \\makebox[0.18\\textwidth]{{\\scriptsize {}}}\n".format("source")
    latex_str+="    \\makebox[0.18\\textwidth]{{\\scriptsize {}}}\n".format("driving")
    for m in methods:
        latex_str+="    \\makebox[0.18\\textwidth]{{\\scriptsize {}}}\n".format(m)
    latex_str=latex_str[:-1]+ r'\\'+'\n'
    img_str = "    \\includegraphics[width=0.11\\textwidth]{{image/chap04/experiment_src_driv/{}}}\n"
    generator_dict, kp_detector_dict = load_checkpoints(config, methods)
    for i in tqdm(range(num)):
        source_image, dest_image, default_fname = random_img_pair(dataset,mode,seed)
        image = visualize_comparison(generator_dict, kp_detector_dict, methods, source_image, dest_image)
        imageio.imsave(default_fname if path is None else path, image)
        latex_str+=img_str.format(default_fname)
        if i!=num-1:
            latex_str+=r'    \\'+'\n'
    latex_str+="    \\caption{%s}\n    \\label{%s}"%(caption,"fig:"+label)
    latex_str+=('\n'+r"\end{figure}")
    return latex_str


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode",choices=["rec","anim"], default="rec", help="reconstruction or animation")
    parser.add_argument("--format",choices=["row","table"], required=True, help="row: show images for different methods in one row. table: show images for different sources and driving images in tables.")
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--methods", default=[], type=lambda x: eval(x),
                        help="Names of the methods comma separated.")
    parser.add_argument("--path", default=None, help="path to result image")
    parser.add_argument("--num", default=1, type=int, help="number of image")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    dataset = FramesDataset(is_train=False, **config['dataset_params'])
    if opt.format=="row":
        print(gen_compare2(config,dataset,opt.methods,opt.num,opt.path,opt.mode,opt.seed))
    elif opt.format=="table":
        for method in opt.methods:
            gen_table(config, dataset, 4, 4, method, opt.seed)
