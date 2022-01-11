import numpy as np
from pytorch_msssim import MS_SSIM  # https://github.com/VainF/pytorch-msssim
from ignite.metrics.gan import FID
import lpips, torch
import pandas as pd
import threading
from tqdm import tqdm


class evaluator:
    def __init__(self, kp_detector, generator, metrics_lst, size):
        self.kp_detector = kp_detector
        self.generator = generator
        self.loss_fn = lpips.LPIPS(net='vgg', verbose=False)
        self.ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)
        self.m = FID(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.metrics_lst = metrics_lst
        self.metric_func = {
            'l1': self.L1_norm,
            'LPIPS': self.LPIPS,
            'PSNR': self.MSE,
            'MS-/SSIM': self.MS_SSIM,
            'FID': self._FID,
            'AKD': self._AKD
        }
        self.video_cnt = 0
        self.result = np.zeros([size, len(metrics_lst)])
        self.metric_index = {v: k for k, v in dict(enumerate(metrics_lst)).items()}
        self.FIDpos = self.metrics_lst.index("FID") if "FID" in self.metrics_lst else None
        self.AKDpos = self.metrics_lst.index("AKD") if "AKD" in self.metrics_lst else None

    def generate(self, source, driving_frame):
        source_tensor = torch.tensor(source[np.newaxis].astype(np.float32)).cuda()
        with torch.no_grad():
            kp_source = self.kp_detector(source_tensor)
            kp_driving = self.kp_detector(torch.tensor(driving_frame[np.newaxis].astype(np.float32)).cuda())
            out = self.generator(source_tensor, kp_source=kp_source, kp_driving=kp_driving)
            kp_prediction = self.kp_detector(out['prediction'])
        return out['prediction'].data.cpu().numpy()[0], kp_driving['value'].squeeze_().cpu().numpy(), kp_prediction[
            'value'].squeeze_().cpu().numpy()

    def L1_norm(self, src, pred):
        self.result[self.video_cnt, self.metric_index['l1']] += np.sum(np.abs(src - pred)) / np.size(src)

    def LPIPS(self, src, pred):
        with torch.no_grad():
            self.result[self.video_cnt, self.metric_index['LPIPS']] += float(
                self.loss_fn.forward(torch.tensor(src, dtype=torch.float32),
                                     torch.tensor(pred, dtype=torch.float32)))

    def MSE(self, src, pred):
        self.result[self.video_cnt, self.metric_index['PSNR']] = np.sum((src - pred) ** 2) / np.size(src)

    def MS_SSIM(self, src, pred):
        self.result[self.video_cnt, self.metric_index['MS-/SSIM']] = float(
            1 - self.ms_ssim_module(255.0 * torch.tensor(src, dtype=torch.float32).unsqueeze_(0),
                                    255.0 * torch.tensor(pred, dtype=torch.float32).unsqueeze_(0)))

    @staticmethod
    def _FID(src, pred):
        """
        Placeholder function
        """
        return 0

    @staticmethod
    def _AKD(src, pred):
        """
        Placeholder function
        """
        return 0

    def FID(self, src, pred):
        """
        Video quality. Input must be video
        """
        self.m.reset()
        self.m.update((torch.tensor(pred, dtype=torch.float32) * 255.0, torch.tensor(src, dtype=torch.float32) * 255.0))
        self.result[self.video_cnt, self.metric_index['FID']] = float(self.m.compute())

    def AKD(self, kp_driving, kp_prediction):
        with torch.no_grad():
            self.result[self.video_cnt, self.AKDpos] = np.average(
                np.sqrt(np.sum((kp_driving - kp_prediction) ** 2, axis=1)))
            
    def evaluate(self, source_img, driving_video):
        length = len(driving_video)
        pred_video = []
        for driving_frame in tqdm(driving_video):
            pred_frame, kp_driving, kp_prediction = self.generate(source_img, driving_frame)
            pred_video.append(pred_frame)
            for metric in self.metrics_lst:
                t = threading.Thread(target=self.metric_func[metric], args=(driving_frame, pred_frame,))
                t.start()
                t.join()
            if not self.AKDpos is None:
                t = threading.Thread(target=self.AKD, args=(kp_driving, kp_prediction,))
                t.start()
                t.join()
        self.result[self.video_cnt] /= length
        if not self.FIDpos is None:
            t = threading.Thread(target=self.FID, args=(driving_video, pred_video,))
            t.start()
            t.join()
        self.video_cnt += 1

    def get_res_list(self):
        return [res / self.video_cnt for res in self.result]

    def get_res_pd(self):
        df = pd.DataFrame(self.result, columns=self.metrics_lst)
        df = df.append(df.mean(axis=0), ignore_index=True)
        return df

    def save_res(self, filename='1.csv'):
        self.get_res_pd().to_csv(filename)
