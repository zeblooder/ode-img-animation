import torch
from torch.nn.modules.module import Module
from torch.nn import functional as F
class Resample2d(Module):

    def __init__(self, kernel_size=2, dilation=1, sigma=5 ):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = torch.tensor(sigma, dtype=torch.float).cuda()

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        # sigma = self.sigma.expand(input2.size(0), 1, input2.size(2), input2.size(3)).type(input2.dtype)
        # input2 = torch.cat((input2,sigma), 1)
        # return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.dilation)
        return F.grid_sample(input1_c,input2.permute(0,2,3,1))
