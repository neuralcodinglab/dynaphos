import logging
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from typing import Optional, Tuple, Union, Iterable


def get_deg2pix_coeff(run_params: dict) -> float:
    view_angle = run_params['view_angle']
    resolution = run_params['resolution']
    deg2pix = resolution[0] / view_angle
    logging.debug(f"Displaying {view_angle} degrees of vision in a resolution "
                  f"of {resolution}.")
    logging.debug(f"One degree is equivalent to {deg2pix} pixels.")
    return deg2pix


def calculate_dpi(params: dict) -> float:
    w_pixels = params['display']['screen_resolution'][0]
    h_pixels = params['display']['screen_resolution'][1]
    diagonal = params['display']['screen_diagonal']
    w_inches = (diagonal ** 2 / (1 + h_pixels ** 2 / w_pixels ** 2)) ** 0.5
    dpi = round(w_pixels / w_inches)
    return dpi


def display_real_size(params: dict, image: np.ndarray):
    mm_per_degree = \
        params['display']['dist_to_screen'] * np.tan(2 * np.pi / 360)
    view_angle = params['run']['view_angle']
    resolution = params['run']['resolution']
    aspect_ratio = resolution[0] / resolution[1]

    mm = 0.1 / 2.54
    fig_width = mm_per_degree * view_angle * mm
    fig_height = mm_per_degree * (view_angle / aspect_ratio) * mm
    dpi = calculate_dpi(params)
    logging.debug(f"Display sizes: {fig_width}, {fig_height} | dpi: {dpi}")

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, origin='lower')
    plt.show()


def load_coordinates_from_yaml(path: str, n_coordinates: Optional[int] = None,
                               rng: Optional[np.random.Generator] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'r') as f:
        coordinates = yaml.load(f, Loader=yaml.FullLoader)
        x = np.array(coordinates['x'])
        y = np.array(coordinates['y'])

    if n_coordinates:
        if rng is None:
            rng = np.random.default_rng()
        sample = rng.choice(len(x), n_coordinates)
        x = x[sample]
        y = y[sample]

    return x, y


def load_params(path: str) -> dict:
    with open(path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def to_tensor(x: Union[int, float, np.ndarray], **data_kwargs) -> torch.Tensor:
    return torch.tensor(x, **data_kwargs)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


def get_data_kwargs(params: dict) -> dict:
    dtype = getattr(torch, params['run']['dtype'])
    gpu = params['run']['gpu']
    device = 'cpu' if not torch.cuda.device_count() or gpu is None \
        else f'cuda:{gpu}'
    return dict(device=device, dtype=dtype)


def cartesian_to_complex(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + 1j * y


def polar_to_complex(r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return r * np.exp(1j * phi)


def complex_to_cartesian(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.real(z), np.imag(z)


def complex_to_polar(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.abs(z), np.angle(z)


def cartesian_to_polar(x: np.ndarray, y: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    return r, phi


def get_truncated_normal(size: Union[int, Iterable], mean: float, sd: float,
                         low: Optional[float] = 0.,
                         upp: Optional[float] = 1e-4) -> np.ndarray:
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)


class Map:
    def __init__(self,
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 z: Optional[np.ndarray] = None,
                 r: Optional[np.ndarray] = None,
                 phi: Optional[np.ndarray] = None):
        assert ((x is not None and y is not None) ^ (z is not None) ^
                (r is not None and phi is not None)), "Invalid arguments."

        self._x = x
        self._y = y
        self._z = z
        self._r = r
        self._phi = phi

        if self._x is not None and self._y is not None:
            self._z = cartesian_to_complex(self._x, self._y)
            self._r, self._phi = complex_to_polar(self._z)
        elif self._z is not None:
            self._x, self._y = complex_to_cartesian(self._z)
            self._r, self._phi = complex_to_polar(self._z)
        elif self._r is not None and self._phi is not None:
            self._z = polar_to_complex(self._r, self._phi)
            self._x, self._y = complex_to_cartesian(self._z)

    def __len__(self):
        if self._x is None:
            return 0
        return len(self._x)

    @property
    def polar(self):
        return self._r, self._phi

    @property
    def cartesian(self):
        return self._x, self._y

    @property
    def complex(self):
        return self._z

    def use_subset(self, indexes: np.ndarray):
        self._x = self._x[indexes]
        self._y = self._y[indexes]
        self._z = self._z[indexes]
        self._r = self._r[indexes]
        self._phi = self._phi[indexes]


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid for brightness saturation and thresholding psychometric curves.
    """
    return torch.div(1, 1 + torch.exp(-x))


def print_stats(stat_name: str, stat: torch.Tensor, verbose=False):
    if verbose:
        msg = f"""{stat_name}:
            size:   {stat.size()}
            min:    {stat.min():.2E}
            max:    {stat.max():.2E}
            mean:   {stat.mean():.2E}
            std:    {stat.std():.2E}"""
        logging.debug(msg)
