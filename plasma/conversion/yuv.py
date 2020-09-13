# 
#   Plasma
#   Copyright (c) 2020 Homedeck, LLC.
#

from torch import tensor, Tensor

def rgb_to_yuv (input: Tensor) -> Tensor:
    """
    Convert an array of RGB pixels to YUV.
    The shape of the input tensor is preserved.

    Parameters:
        input (Tensor): Input RGB pixel array with shape (N,3,...) in [-1., 1.].

    Returns:
        Tensor: YUV pixel array with shape (N,3,...) in [0., 1.]
    """
    RGB_TO_YUV = tensor([
        [0.2126, 0.7152, 0.0722],
        [-0.09991, -0.33609, 0.436],
        [0.615, -0.55861, -0.05639]
    ]).float().to(input.device)
    input = (input + 1.) / 2.
    rgb_colors = input.flatten(start_dim=2)
    yuv_colors = RGB_TO_YUV.matmul(rgb_colors)
    yuv = yuv_colors.view_as(input)
    return yuv

def yuv_to_rgb (input: Tensor) -> Tensor:
    """
    Convert an array of YUV pixels to RGB.
    The shape of the input tensor is preserved.

    Parameters:
        input (Tensor): Input YUV pixel array with shape (N,3,...) in [0., 1.].

    Returns:
        Tensor: RGB pixel array with shape (N,3,...) in [-1., 1.]
    """
    YUV_TO_RGB = tensor([
        [1., 0., 1.28033],
        [1., -0.21482, -0.38059],
        [1., 2.12798, 0.]
    ]).float().to(input.device)
    yuv_colors = input.flatten(start_dim=2)
    rgb_colors = YUV_TO_RGB.matmul(yuv_colors)
    rgb = rgb_colors.view_as(input)
    rgb = 2.0 * rgb - 1.0
    rgb = rgb.clamp(min=-1., max=1.)
    return rgb

def rgb_to_luminance (input: Tensor) -> Tensor:
    """
    Convert an array of RGB pixels to luminance.

    Parameters:
        input (Tensor): Input RGB pixel array with shape (N,3,...) in [-1., 1.]

    Returns:
        Tensor: Luminance pixel array with shape (N,1,...) in [-1., 1.]
    """
    yuv = rgb_to_yuv(input)
    luminance = yuv[:,:1,...]
    luminance = luminance * 2. - 1.
    return luminance