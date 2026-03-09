import torch

import random

# def random_crop(lq, gt, random_crop_ratio):
#     # lq: low quality image, type: Tensor
#     h, w = lq.shape[1], lq.shape[2]
#     crop_h, crop_w = int(h * random_crop_ratio[0]), int(w * random_crop_ratio[1])
#     assert crop_h == crop_w, "Image crop area should be a square, but got ({}, {})".format(crop_h, crop_w)
#     crop = crop_h

#     rr = random.randint(0, h - crop)
#     cc = random.randint(0, w - crop)

#     lq = lq[:, rr:rr+crop, cc:cc+crop]
#     gt = gt[:, rr:rr+crop, cc:cc+crop]

#     return lq, gt

def random_crop(lq, gt, crop_size: int = 256):
    """
    通用化固定尺寸随机正方形裁剪（支持任意原始图像尺寸）
    Args:
        lq: 低质量图像 Tensor (C, H, W)
        gt: 高质量图像 Tensor (C, H, W)
        crop_size: 训练参数指定的裁剪尺寸（正方形，默认256）
    Returns:
        裁剪后的lq和gt（尺寸为 C × crop_size × crop_size）
    """
    # 获取图像的高和宽（Tensor格式：C, H, W）
    h, w = lq.shape[1], lq.shape[2]
    
    # 若原始图像小于裁剪尺寸：先等比例resize到裁剪尺寸（避免裁剪越界）
    if h < crop_size or w < crop_size:
        # 计算resize比例（取最大比例，确保至少覆盖裁剪尺寸）
        scale = max(crop_size / h, crop_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        # 使用torch.nn.functional插值resize
        lq = torch.nn.functional.interpolate(
            lq.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        gt = torch.nn.functional.interpolate(
            gt.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        # 更新尺寸
        h, w = new_h, new_w
    
    # 随机生成裁剪起始坐标（确保裁剪区域不越界）
    rr = random.randint(0, h - crop_size)
    cc = random.randint(0, w - crop_size)

    # 执行裁剪（最终输出固定为 crop_size × crop_size）
    lq_cropped = lq[:, rr:rr+crop_size, cc:cc+crop_size]
    gt_cropped = gt[:, rr:rr+crop_size, cc:cc+crop_size]

    return lq_cropped, gt_cropped

def data_augment_lowlevel_torch(input, mode):
    '''
    input: image [3, h, w] may be input or target
    mode: 0, 1, 2, 3, ..., 7 totall 8
    '''

    # Data Augmentations
    if mode == 0:
        output = input
    elif mode == 1:
        output = input.flip(1)
    elif mode == 2:
        output = input.flip(2)
    elif mode == 3:
        output = torch.rot90(input, dims=(1, 2))
    elif mode == 4:
        output = torch.rot90(input, dims=(1, 2), k=2)
    elif mode == 5:
        output = torch.rot90(input, dims=(1, 2), k=3)
    elif mode == 6:
        output = torch.rot90(input.flip(1), dims=(1, 2))
    elif mode == 7:
        output = torch.rot90(input.flip(2), dims=(1, 2))
    else:
        raise NotImplementedError()

    return output