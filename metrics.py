import numpy as np
from pytorch_msssim import MS_SSIM #https://github.com/VainF/pytorch-msssim
from ignite.metrics.gan import FID
import lpips, torch
import pandas as pd
class evaluator:
    def __init__(self,kp_detector,metrics_lst,size):
        self.kp_detector=kp_detector
        self.loss_fn = lpips.LPIPS(net='vgg', verbose=False)
        self.ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
        self.m = FID()
        self.metrics_lst=metrics_lst
        self.metric_func={
        'l1':self.L1_norm,
        'LPIPS':self.LPIPS,
        'PSNR' :self.MSE,
        'MS-/SSIM':self.MS_SSIM,
        'FID':self._FID,
        'AKD':self.AKD
    }
        self.video_cnt=0
        self.result=np.zeros([size,len(metrics_lst)])
        self.metric_index={v:k for k,v in dict(enumerate(metrics_lst)).items()}
        self.FIDpos=self.metrics_lst.index("FID")if "FID" in self.metrics_lst else None
    @staticmethod
    def L1_norm(src,pred):
        return np.sum(np.abs(src-pred))/np.size(src)

    def LPIPS(self,src,pred):
        with torch.no_grad():
            return float(self.loss_fn.forward(torch.tensor(src,dtype=torch.float32), torch.tensor(pred,dtype=torch.float32)))
    @staticmethod
    def MSE(src,pred):
        return np.sum((src-pred)**2)/np.size(src)


    def MS_SSIM(self,src,pred):
        return float(1 - self.ms_ssim_module(torch.tensor(255*src,dtype=torch.float32).unsqueeze_(0), torch.tensor(255*pred,dtype=torch.float32).unsqueeze_(0)))

    @staticmethod
    def _FID(src,pred):
        """
        Placeholder function
        """
        return 0

    def FID(self,src,pred):
        """
        Video quality. Input must be video
        """
        self.m.reset()
        self.m.update((torch.tensor(255*pred,dtype=torch.float32), torch.tensor(255*src,dtype=torch.float32)))
        return self.m.compute()

    def AKD(self, src, pred):
        with torch.no_grad():
            src = torch.tensor(src[np.newaxis].astype(np.float32))
            kp_source = self.kp_detector(src)
            pred = torch.tensor(np.array(pred)[np.newaxis].astype(np.float32))
            kp_prediction = self.kp_detector(pred)
            return np.average(np.sqrt(np.sum((kp_source['value'].squeeze_().cpu().numpy()-kp_prediction['value'].squeeze_().cpu().numpy())**2,axis=1)))

    def evaluate(self,src_video,pred_video):
        assert len(src_video)==len(pred_video)
        length=len(src_video)
        metrics=[0]*len(self.metrics_lst)
        for src_frame, pred_frame in zip(src_video,pred_video):
            metrics=[self.metric_func[metric](src_frame,pred_frame) for metric in self.metrics_lst]
        metrics=[data/length for data in metrics]
        if not self.FIDpos is None:
            metrics[self.FIDpos]=float(self.FID(src_video,pred_video))

        self.result[self.video_cnt,[self.metric_index[m] for m in self.metrics_lst]]=metrics
        self.video_cnt+=1

    def get_res_list(self):
        return [res/self.video_cnt for res in self.result]

    def get_res_pd(self):
        df=pd.DataFrame(self.result, columns=self.metrics_lst)
        df=df.append(df.mean(axis=0),ignore_index=True)
        return df