import logging
import torch
from typing import Optional, Tuple, Callable, Union

import numpy as np

from dynaphos.utils import (Map, cartesian_to_complex, polar_to_complex,
                            complex_to_polar)


def get_visual_field_coordinates_from_cortex(
        params: dict, coordinates_cortex: Optional[Map] = None,
        rng: Optional[np.random.Generator] = None) -> Map:
    """Map electrode locations from cortex to visual field.

    :param params: Parameters for visuotopic model.
    :param coordinates_cortex: Locations of electrodes on a cortex map.
    :param rng: Numpy random number generator.
    :return: Location of electrodes / phosphenes in the visual field.
    """
    if coordinates_cortex is None:
        coordinates_cortex = get_cortex_coordinates_grid(params, 32, 32,
                                                         x_max=40)

    x, y = coordinates_cortex.cartesian
    x, y = add_noise(x, y, params['noise_scale'], rng)
    x, y = add_dropout(x, y, params['dropout_rate'], rng)

    cortex_to_visual_field = get_mapping_from_cortex_to_visual_field(params)
    z = cortex_to_visual_field(cartesian_to_complex(x, y))
    z = remove_out_of_view(z)

    # Flip y-coordinates to account for upside-down orientation of visual
    # field.
    z.imag *= -1

    return Map(z=z)


def get_visual_field_coordinates_from_cortex_full(
        params: dict, coordinates_cortex: Optional[Map] = None,
        rng: Optional[np.random.Generator] = None) -> Map:
    """Initialize phosphene locations in the full field of view.

    :param params: dictionary with the several parameters in subdictionaries.
    :param coordinates_cortex: Visuotopic map of with electrode locations on
        cortex.
    If None, default coordinates will be used.
    :param rng: Numpy random number generator.
    :return: Phosphene locations.
    """
    args = (params, coordinates_cortex, rng)
    r_left, phi_left = get_visual_field_coordinates_from_cortex(*args).polar
    r_right, phi_right = get_visual_field_coordinates_from_cortex(*args).polar
    r = np.concatenate([r_left, r_right])
    phi = np.concatenate([phi_left, np.pi - phi_right])
    return Map(r=r, phi=phi)


def get_visual_field_coordinates_probabilistically(
        params: dict, n_phosphenes: int,
        rng: Optional[np.random.Generator] = None) -> Map:
    """Generate a number of phosphene locations probabilistically.

    :param params: Model parameters.
    :param n_phosphenes: Number of phosphenes.
    :param rng: Numpy random number generator.
    :return: Polar coordinates of n_phosphenes phosphenes.
    """
    if rng is None:
        rng = np.random.default_rng()

    max_r = params['run']['view_angle'] / 2
    valid_ecc = np.linspace(1e-3, max_r, 1000)
    weights = get_cortical_magnification(valid_ecc, params['cortex_model'])

    probs = weights / np.sum(weights)
    r = rng.choice(valid_ecc, size=n_phosphenes, replace=True, p=probs)
    phi = 2 * np.pi * rng.random(n_phosphenes)

    return Map(r=r, phi=phi)


def get_visual_field_coordinates_grid() -> Map:
    ecc_range = np.arange(0, 90, 1)
    ang_range = np.linspace(-np.pi / 2, np.pi / 2, 10)
    r, phi = np.meshgrid(ecc_range, ang_range)
    return Map(r=r.ravel(), phi=phi.ravel())


def get_cortex_coordinates_grid(params: dict, n_electrodes_x: int,
                                n_electrodes_y: int,
                                x_max: Optional[int] = None) -> Map:
    coordinates = get_cortex_coordinates_default(params)
    x, y = coordinates.cartesian
    x_min, x_max = np.min(x), x_max or np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    xrange = np.linspace(x_min, x_max, n_electrodes_x)
    yrange = np.linspace(y_min, y_max, n_electrodes_y)
    x, y = np.meshgrid(xrange, yrange)

    return Map(x.ravel(), y.ravel())


def get_cortex_coordinates_default(params: dict) -> Map:
    """Generate cortical map.

    :return: Cortical coordinates.
    """
    coordinates_visual_field = get_visual_field_coordinates_grid()
    visual_field_to_cortex = get_mapping_from_visual_field_to_cortex(params)
    z = visual_field_to_cortex(coordinates_visual_field.complex)
    return Map(z=z)


def get_mapping_from_visual_field_to_cortex(params: dict) -> Callable:
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    alpha = params['alpha']
    if mapping_model == 'monopole':
        def f(z): return k * np.log(1 + z / a)
    elif mapping_model == 'dipole':
        def f(z): return k * np.log(b * (z + a) / (a * (z + b)))
    elif mapping_model == 'wedge-dipole':
        def wedge(r, phi): return polar_to_complex(r, alpha * phi)
        def dipole(z): return k * np.log(b * (z + a) / (a * (z + b)))
        def f(z): return dipole(wedge(*complex_to_polar(z)))
    else:
        raise NotImplementedError
    return f


def get_mapping_from_cortex_to_visual_field(params: dict) -> Callable:
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    alpha = params['alpha']
    if mapping_model == 'monopole':
        def f(w): return a * np.exp(w / k) - a
    elif mapping_model == 'dipole':
        def f(w):
            e = np.exp(w / k)
            return a * b * (e - 1) / (b - a * e)
    elif mapping_model == 'wedge-dipole':
        def wedge_inverse(z):
            r, phi = complex_to_polar(z)
            return polar_to_complex(r, phi / alpha)

        def dipole_inverse(w):
            e = np.exp(w / k)
            return a * b * (e - 1) / (b - a * e)

        def f(w): return wedge_inverse(dipole_inverse(w))
    else:
        raise NotImplementedError
    return f


def get_cortical_magnification(
        r: Union[np.ndarray, torch.Tensor],
        params: dict) -> Union[np.ndarray, torch.Tensor]:
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    if mapping_model == 'monopole':
        return k / (r + a)
    if mapping_model in ['dipole', 'wedge-dipole']:
        return k * (1 / (r + a) - 1 / (r + b))
    raise NotImplementedError


def remove_out_of_view(z: np.ndarray) -> np.ndarray:
    r, phi = complex_to_polar(z)

    z = z[(r >= 0) & (r <= 90) & (phi > -np.pi / 2) & (phi < np.pi / 2)]

    logging.info(f"Removed {len(r) - len(z)} of {len(r)} phosphene locations.")

    return z


def add_noise(x: np.ndarray, y: np.ndarray, noise_scale: Optional[float] = 0.,
              rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray,
                                                                  np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(scale=noise_scale, size=(2, len(x)))

    return x + noise[0], y + noise[1]


def add_dropout(x: np.ndarray, y: np.ndarray,
                dropout_rate: Optional[float] = 0.,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    active = rng.choice(np.arange(n), int(n * (1 - dropout_rate)),
                        replace=False)

    return x[active], y[active]
