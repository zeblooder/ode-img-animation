import numpy as np
import torch,sys
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from .dense_motion import DenseMotionNetwork
from .odenet import ODENet
from .appearance_flow import FlowGen
class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, integration_time, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None
        self.ode = ODENet(integration_time)
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        self.flow_param = {'input_dim': 256, 'dim': 64, 'n_res': 2, 'activ': 'relu',
                           'norm_conv': 'ln', 'norm_flow': 'in', 'pad_type': 'reflect', 'use_sn': False}
        self.appearance_flow=FlowGen(**self.flow_param)

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        num_down_blocks+=1
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def forward(self, source_image, kp_driving, kp_source):
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}

        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']

        torch.cuda.empty_cache()

        # Encoding (downsampling) part, out?????????F_S
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        torch.cuda.empty_cache()

        # Transforming feature representation according to deformation and occlusion
        ori_out = self.deform_input(out, deformation)  # F_{SD}
        if self.ode is not None:
            dense_motion = self.ode(deformation) # ??????ODE,??????F_{S->D} 1*64*64*2

            out = self.deform_input(out, dense_motion) # F_{SD}

        if self.appearance_flow is not None:
            F_app, flow_maps = self.appearance_flow(out) # F_{SD}

        if occlusion_map is not None:
            if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear', align_corners=True)
            F_app = F_app * occlusion_map
            ori_out = ori_out * occlusion_map
        out=torch.cat([ori_out, F_app],dim=1)
        output_dict["deformed"] = self.deform_input(source_image, deformation)

        torch.cuda.empty_cache()
        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)

        out = F.interpolate(out,scale_factor=0.5,recompute_scale_factor=True)
        output_dict["prediction"] = out

        return output_dict
