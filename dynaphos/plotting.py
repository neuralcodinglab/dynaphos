from typing import Optional

from matplotlib import pyplot as plt

from dynaphos.cortex_models import get_cortex_coordinates_default, get_cortex_coordinates_grid


def plot_coordinates(params: dict, n_electrodes_x: int, n_electrodes_y: int,
                     x_max: Optional[int] = None):
    coordinates_cortex = get_cortex_coordinates_default(params)
    grid_cortex = get_cortex_coordinates_grid(params, n_electrodes_x,
                                              n_electrodes_y, x_max)
    plt.scatter(*coordinates_cortex.cartesian, c='r')
    plt.scatter(*grid_cortex.cartesian)
    plt.xlabel('Cortical distance (mm)')
    plt.ylabel('Cortical distance (mm)')
    plt.show()
