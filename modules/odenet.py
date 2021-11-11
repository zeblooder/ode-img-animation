import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint, odeint
from .util import SameBlock2d
# parser = argparse.ArgumentParser()
# parser.add_argument('--tol', type=float, default=1e-3)
# parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
# parser.add_argument('--nepochs', type=int, default=160)
# parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
# parser.add_argument('--lr', type=float, default=0.1)
# parser.add_argument('--batch_size', type=int, default=128)
# parser.add_argument('--test_batch_size', type=int, default=1000)
#
# parser.add_argument('--save', type=str, default='./experiment1')
# parser.add_argument('--debug', action='store_true')
# parser.add_argument('--gpu', type=int, default=0)
# args = parser.parse_args()
#


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, adjoint=True,tol=1e-3):

        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol=tol
        self.odeint= odeint_adjoint if adjoint else odeint

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]



class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


# def accuracy(model, dataset_loader):
#     total_correct = 0
#     for x, y in dataset_loader:
#         x = x.to(device)
#         y = one_hot(np.array(y.numpy()), 10)
#
#         target_class = np.argmax(y, axis=1)
#         predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
#         total_correct += np.sum(predicted_class == target_class)
#     return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class ODENet(nn.Module):
    def __init__(self):
        super(ODENet, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # downsampling_layers = [
        #     nn.Conv2d(1, 64, 3, 1),
        #     norm(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 4, 2, 1),
        #     norm(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 4, 2, 1),
        # ]
        feature_layers = [ODEBlock(ODEfunc(64))]
        # fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), (64,1,3)]

        self.model = nn.Sequential(*feature_layers).to(device)

    def forward(self, x):
        return self.model(x)

# if __name__ == '__main__':
#
#     makedirs(args.save)
#
#
#
#
#
#     print('Number of parameters: {}'.format(count_parameters(model)))
#
#     criterion = nn.CrossEntropyLoss().to(device)
#
#     train_loader, test_loader, train_eval_loader = get_mnist_loaders(
#         args.data_aug, args.batch_size, args.test_batch_size
#     )
#
#     data_gen = inf_generator(train_loader)
#     batches_per_epoch = len(train_loader)
#
#     lr_fn = learning_rate_with_decay(
#         args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
#         decay_rates=[1, 0.1, 0.01, 0.001]
#     )
#
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
#
#     best_acc = 0
#     batch_time_meter = RunningAverageMeter()
#     end = time.time()
#
#     for itr in range(args.nepochs * batches_per_epoch):
#
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr_fn(itr)
#
#         optimizer.zero_grad()
#         x, y = data_gen.__next__()
#         x = x.to(device)
#         y = y.to(device)
#         logits = model(x)
#         loss = criterion(logits, y)
#
#         loss.backward()
#         optimizer.step()
#
#         batch_time_meter.update(time.time() - end)
#         end = time.time()
#
#         if itr % batches_per_epoch == 0:
#             with torch.no_grad():
#                 train_acc = accuracy(model, train_eval_loader)
#                 val_acc = accuracy(model, test_loader)
#                 if val_acc > best_acc:
#                     torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
#                     best_acc = val_acc
#                 print(
#                     "Epoch {:04d} | Time {:.3f} ({:.3f})"
#                     "Train Acc {:.4f} | Test Acc {:.4f}".format(
#                         itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, train_acc, val_acc
#                     )
#                 )
