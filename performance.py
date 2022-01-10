import os
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
import torch
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
    return imgLst


def performance(generator, kp_detector, dataset, metrics, source_image=None):
    if source_image is None:
        rand_video = dataset.__getitem__(np.random.randint(len(dataset)))['video']
        source_image = rand_video[:, np.random.randint(rand_video.shape[1]), :, :]
    else:
        source_image = imageio.imread(source_image)
    if torch.cuda.is_available():
        # generator = DataParallelWithCallback(generator)
        generator = generator.cuda()
        # kp_detector = DatarallelWithCallback(generator)
        kp_detector = kp_detector.cuda()

    generator.eval()
    kp_detector.eval()
    e = evaluator(kp_detector, generator, metrics, len(dataset))

    for it, x in tqdm(enumerate(dataset.videos)):
        driving_img_lst = video2imgLst(os.path.join(dataset.root_dir, x))
        e.evaluate(source_image, driving_img_lst)
    e.save_res()
