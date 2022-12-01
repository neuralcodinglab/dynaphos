import sys

import pandas as pd
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import load_params, load_coordinates_from_yaml, Map, \
    get_data_kwargs
from dynaphos.cortex_models import \
    get_visual_field_coordinates_from_cortex_full


def get_random(resolution, num_phosphenes, num_frames=1000):
    params = load_params('../config/params.yaml')
    data_kwargs = get_data_kwargs(params)
    rng = np.random.default_rng(seed=params['run']['seed'])
    params['run']['resolution'] = list(resolution)
    shape = [num_frames] + params['run']['resolution']
    coordinates_cortex = load_coordinates_from_yaml(
        '../config/grid_coords_dipole_valid.yaml', num_phosphenes, rng)
    coordinates_cortex = Map(*coordinates_cortex)
    coordinates_visual_field = get_visual_field_coordinates_from_cortex_full(
        params['cortex_model'], coordinates_cortex, rng)
    simulator = GaussianSimulator(params, coordinates_visual_field, rng)
    frames = torch.mul(torch.rand(shape, **data_kwargs), 255)

    stim_sequence = []
    for frame in frames:
        stim_sequence.append(simulator.sample_stimulus(frame))
    return simulator, stim_sequence

def run_simulation(simulator, stim_sequence):
    for stimulus in stim_sequence:
        simulator(stimulus)
    

def run_single(n=5):
    simulator, stim_sequence = get_random([256, 256], 100, 1000)
    timer = timeit.Timer(lambda: run_simulation(simulator, stim_sequence))
    result = timer.timeit(n)
    print(result / n)


def run_sweep():
    num_frames = 1000
    num_repetitions = 5
    num_phosphenes = []
    resolutions = []
    fps = []
    for _ in tqdm.tqdm(range(num_repetitions), 'Repetition', leave=False):
        for m in tqdm.tqdm([128, 256, 512, 1024], 'Phosphenes', leave=False):
            for k in tqdm.tqdm([64, 128, 256, 512], 'Resolution', leave=False):
                simulator, stim_sequence = get_random([k, k], m, num_frames)
                timer = timeit.Timer(lambda: run_simulation(simulator, stim_sequence))
                t = timer.timeit(1)
                num_phosphenes.append(m)
                resolutions.append(k)
                fps.append(num_frames / t)
    data = pd.DataFrame(dict(num_phosphenes=num_phosphenes,
                             resolution=resolutions, fps=fps))
    summary = data.groupby(['resolution', 'num_phosphenes']).mean()
    print(summary)
    pd.to_pickle(data, 'benchmark.pkl')
    sns.relplot(data=data, x='num_phosphenes', y='fps', hue='resolution',
                style='resolution', kind='line', markers=True)
    plt.savefig('benchmark.png')


if __name__ == '__main__':
    run_sweep()
    # run_single()

    sys.exit()
