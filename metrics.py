import numpy as np
from pytorch_msssim import MS_SSIM #https://github.com/VainF/pytorch-msssim
from ignite.metric.gan import FID

def L1_norm(src,pred):
    return np.norm(src-pred,ord=1)

def LPIPS(src,pred):
    return np.norm(src-pred,ord=1)

def MSE(src,pred):
    return np.norm(src-pred)

def MS_SSIM(src,pred):
    ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
    return 1 - ms_ssim_module(src, pred)

def FID(src,pred):
    m = FID()
    m.update((pred, src))
    return m.compute()

def AKD(src,pred):
    raise NotImplementedError

metric_func={
    'l1': L1_norm,
    'LPIPS': LPIPS,
    'PSNR' : MSE,
    'MS-/SSIM':MS_SSIM,
    'FID':FID,
    'AKD':AKD
}