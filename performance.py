import os

import imageio
import numpy as np
import torch
from skimage.transform import resize
from metrics import evaluator


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
    return imgLst[:2]


def performance(generator, kp_detector, dataset, metrics, result_table, specified_source, source_image=None):
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
        source_image = imageio.imread(source_image)

    if torch.cuda.is_available():
        # generator = DataParallelWithCallback(generator)
        generator = generator.cuda()
        # kp_detector = DatarallelWithCallback(generator)
        kp_detector = kp_detector.cuda()

    generator.eval()
    kp_detector.eval()
    e = evaluator(kp_detector, generator, metrics, len(new_dataset))

    for it, x in enumerate(new_dataset[:2]):
        driving_img_lst = video2imgLst(os.path.join(dataset.root_dir, x))
        e.evaluate(source_image, driving_img_lst)
    e.save_res(result_table)
