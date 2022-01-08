import matplotlib

matplotlib.use('Agg')
import os, sys
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
import torch
from sync_batchnorm import DataParallelWithCallback
from logger import Logger, Visualizer
from torch.utils.data import DataLoader

from animate import normalize_kp
from metrics import evaluator
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,5,7'

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 2, 1, 3, 4)
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

            predictions.append(out['prediction'].data.cpu().numpy()[0])
    return predictions

def video2imgLst(video,transpose=True):
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']
    imgLst = []
    try:
        for im in reader:
            imgLst.append(im)
    except RuntimeError:
        pass
    reader.close()
    if transpose:
        imgLst = [resize(frame, (256, 256))[..., :3].transpose(2,0,1) for frame in imgLst]
    else:
        imgLst = [resize(frame, (256, 256))[..., :3] for frame in imgLst]
    return imgLst[:2]

def gen_video(source_image,driving_video,generator,kp_detector,relative=True, adapt_scale=True, cpu=False):
    driving_video=video2imgLst(driving_video)
    source_image = resize(source_image, (256, 256))[..., :3]
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative,
                                     adapt_movement_scale=adapt_scale, cpu=cpu)
    return driving_video, predictions

def performance(generator, kp_detector, dataset, metrics,source_image=None):

    if source_image is None:
        rand_video = dataset.__getitem__(np.random.randint(len(dataset)))['video']
        source_image = rand_video[:,np.random.randint(rand_video.shape[1]),:,:]
    else:
        source_image = imageio.imread(source_image)
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    e=evaluator(kp_detector,metrics,len(dataset))

    for it, x in tqdm(enumerate(dataset.videos)):
        src, pred=gen_video(source_image,os.path.join(dataset.root_dir,x),generator,kp_detector)
        e.evaluate(src,pred)
    e.get_res_pd().to_csv('1.csv')