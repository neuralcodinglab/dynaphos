import torch
from typing import Optional, Union

import cv2
import numpy as np


def canny_processor(frame: np.ndarray, threshold_low: float,
                    threshold_high: float) -> np.ndarray:
    return cv2.Canny(frame, threshold_low, threshold_high)


def sobel_processor(frame: np.ndarray) -> np.ndarray:
    kwargs = dict(ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.Sobel(frame, cv2.CV_16S, 1, 0, **kwargs)
    grad_y = cv2.Sobel(frame, cv2.CV_16S, 0, 1, **kwargs)
    xy = np.stack([grad_x, grad_y])
    grad = np.linalg.norm(xy, axis=0)
    return grad


def to_n_dim(image: Union[np.ndarray, torch.Tensor], n: Optional[int] = 3
             ) -> Union[np.ndarray, torch.Tensor]:
    while image.ndim < n:
        if isinstance(image, torch.Tensor):
            image = torch.unsqueeze(image, 0)
        else:
            image = np.expand_dims(image, 0)
    return image


def scale_image(image: Union[np.ndarray, torch.Tensor],
                f: Optional[float] = None, use_max: Optional[bool] = False
                ) -> Union[np.ndarray, torch.Tensor]:
    if use_max:
        m = np.max if isinstance(image, np.ndarray) else torch.max
        image = image / m(image)
    if f is not None:
        image = image * f
    return image
