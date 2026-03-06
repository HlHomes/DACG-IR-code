import math
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import pytorch_msssim

_reduction_modes = ['none', 'mean', 'sum']

def psnr_loss(pred, target, weight=None, data_range=1.0, reduction='mean'):

    mse = F.mse_loss(pred, target, reduction='none')

    if weight is not None:
        mse = mse * weight

    mse = mse.mean((1, 2, 3))
    

    mse = torch.clamp(mse, min=1e-10)
    psnr_val = 10 * torch.log10((data_range ** 2) / mse)

    if reduction == 'mean':
        return psnr_val.mean()
    elif reduction == 'sum':
        return psnr_val.sum()
    elif reduction == 'none':
        return psnr_val
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Supported: {_reduction_modes}")

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        psnr_val = psnr_loss(pred, target, weight, self.data_range, self.reduction)
        return self.loss_weight * (-psnr_val)

def SSIM_loss(pred_img, real_img, data_range):

    return pytorch_msssim.ssim(pred_img, real_img, data_range=data_range)

class SSIM(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.):
        super(SSIM, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * SSIM_loss(pred, target, self.data_range)

class SSIMloss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.):
        super(SSIMloss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * (1 - SSIM_loss(pred, target, self.data_range))

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):

        target_tensor = None
        device = input.device
        dtype = input.dtype
        if target_is_real:

            create_label = ((self.real_label_var is None) or 
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.full(input.size(), self.real_label, device=device, dtype=dtype)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or 
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.full(input.size(), self.fake_label, device=device, dtype=dtype)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class FocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, epsilon=1e-6, alpha=0.1):
        super(FocalL1Loss, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):

        abs_err = torch.abs(pred - target) / max(self.alpha, 1e-6)

        focal_weight = (torch.log(1 + abs_err + self.epsilon)) ** self.gamma

        focal_l1_loss = focal_weight * abs_err

        return focal_l1_loss.mean()

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=0.1, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):

        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))


        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)

class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l2', reduction='mean'):
        super(EdgeLoss, self).__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')


        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')


        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0)
        self.weight = loss_weight

    def conv_gauss(self, img):

        n_channels = img.size(1)

        kernel = self.kernel.repeat(n_channels, 1, 1, 1).to(img.device)

        pad = kernel.size(2) // 2
        img = F.pad(img, (pad, pad, pad, pad), mode='replicate')
        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current):

        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, pred, target):
        loss = self.criterion(self.laplacian_kernel(pred), self.laplacian_kernel(target))
        return loss * self.weight


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        vgg_pretrained_features = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion='l1', reduction='mean'):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19()
        self.vgg.eval()
        if reduction not in _reduction_modes:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weight = loss_weight

    def forward(self, x, y):

        self.vgg = self.vgg.to(x.device)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):

            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss


class TemperatureScheduler:
    def __init__(self, start_temp, end_temp, total_steps):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_steps = total_steps

    def get_temperature(self, step):

        if step >= self.total_steps:
            return self.end_temp


        cos_inner = math.pi * step / self.total_steps
        temp = self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + math.cos(cos_inner))

        return temp