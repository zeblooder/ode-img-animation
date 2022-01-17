import os

import imageio
import numpy as np
import torch
from skimage.transform import resize
from tqdm import tqdm

from metrics import evaluator
from sync_batchnorm import DataParallelWithCallback


def load_checkpoints(generator, kp_detector, checkpoint_path):
    kp_detector.cuda()
    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()


def video2imgLst(video):
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    imgLst = []
    try:
        for im in reader:
            imgLst.append(im)
    except RuntimeError:
        pass
    reader.close()
    imgLst = [resize(frame, (256, 256))[..., :3].transpose(2, 0, 1) for frame in imgLst]
    return imgLst


def performance(generator, kp_detector, checkpoint_path, dataset, metrics, result_table,
                check_freq=None):
    video_list=dataset.videos
    load_checkpoints(generator, kp_detector, checkpoint_path)
    e = evaluator(kp_detector, generator, metrics, dataset)
    check_freq = int(len(video_list) / 5) if check_freq is None else check_freq

    for it, x in enumerate(tqdm(video_list)):
        driving_img_lst = video2imgLst(os.path.join(dataset.root_dir, x))
        e.evaluate(driving_img_lst)
        if it % check_freq == 0:
            e.save_res(result_table, False)
    e.save_res(result_table, True)