from torch import nn
import torch.nn.functional as F
import torch
from .util import Hourglass, AntiAliasInterpolation2d, kp2grid, make_coordinate_grid, kp2grid


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1)*2,
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):
        spatial_size = source_image.shape[2:]
        grid_driving = kp2grid(kp_driving, spatial_size=spatial_size, kp_variance=self.kp_variance)
        grid_source = kp2grid(kp_source, spatial_size=spatial_size, kp_variance=self.kp_variance)
        heatmap = grid_driving - grid_source

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], 2).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        return heatmap

    def forward(self, source_image, kp_driving, kp_source):
        if self.scale_factor != 1:
            source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict = dict()

        heatmap_representation = self.create_heatmap_representations(source_image,kp_driving,kp_source)
        input = heatmap_representation.view(bs, -1, h, w)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(-1)
        deformation = (heatmap_representation * mask).sum(dim=1)

        out_dict['deformation'] = deformation

        return out_dict
