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


def performance(algorithm, generator, kp_detector, checkpoint_path, dataset, metrics, result_table, specified_source,
                check_freq=None, source_image=None):
    new_dataset = dataset.videos
    if specified_source:
        source_video_index = 0
        rand_video = dataset.__getitem__(source_video_index)['video']
        source_frame_index = 0
        source_image = rand_video[:, source_frame_index, :, :]
        new_dataset.pop(source_video_index)
    elif source_image is None:
        source_video_index = np.random.randint(len(dataset))
        rand_video = dataset.__getitem__(source_video_index)['video']
        source_frame_index = np.random.randint(rand_video.shape[1])
        source_image = rand_video[:, source_frame_index, :, :]
        new_dataset.pop(source_video_index)
    else:
        source_image = resize(imageio.imread(source_image), (256, 256)).transpose([2, 0, 1])

    load_checkpoints(generator, kp_detector, checkpoint_path)
    e = evaluator(kp_detector, generator, metrics, new_dataset)
    check_freq = int(len(new_dataset) / 5) if check_freq is None else check_freq

    for it, x in enumerate(tqdm(new_dataset)):
        driving_img_lst = video2imgLst(os.path.join(dataset.root_dir, x))
        e.evaluate(source_image, driving_img_lst)
        if it % check_freq == 0:
            e.save_res(result_table, False)
    e.save_res(result_table, True)