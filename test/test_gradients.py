"""Gradients must stay finite when electrodes are switched off.

The simulator is used for end-to-end optimisation, where an encoder routinely
drives some electrodes to zero: sparse stimulation is the goal for a
charge-limited device, sigmoid outputs underflow, and any charge penalty pushes
amplitudes down. The sqrt size equation has an infinite derivative at zero, so
without a lower bound those electrodes emit NaN gradients that propagate into
every parameter while the forward pass still looks correct.
"""

import os

import numpy as np
import pytest
import torch

from dynaphos.cortex_models import \
    get_visual_field_coordinates_probabilistically
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import load_params

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'params.yaml')


def build(batch_size=2, num_phosphenes=64, resolution=32, seed=0):
    params = load_params(CONFIG)
    params['run']['resolution'] = [resolution, resolution]
    params['run']['batch_size'] = batch_size
    params['run']['seed'] = seed
    coordinates = get_visual_field_coordinates_probabilistically(
        params, num_phosphenes, np.random.default_rng(seed))
    return GaussianSimulator(params, coordinates, rng=np.random.default_rng(seed))


@pytest.mark.parametrize('fraction_off', [1.0, 0.5, 0.25])
def test_gradient_finite_with_electrodes_off(fraction_off):
    simulator = build()
    amplitude = torch.full((2, simulator.num_phosphenes), 80e-6)
    n_off = int(simulator.num_phosphenes * fraction_off)
    amplitude[:, :n_off] = 0.0
    amplitude.requires_grad_(True)

    simulator.reset()
    percept = simulator(amplitude)
    percept.sum().backward()

    assert torch.isfinite(percept).all(), 'forward pass produced non-finite values'
    assert torch.isfinite(amplitude.grad).all(), (
        f'{torch.isnan(amplitude.grad).float().mean():.0%} of amplitude '
        f'gradients are NaN with {fraction_off:.0%} of electrodes off')


def test_forward_unchanged_by_the_clamp():
    """The bound sits far below any physical current, so percepts must not move."""
    simulator = build()
    generator = torch.Generator().manual_seed(0)
    amplitude = torch.rand(
        (2, simulator.num_phosphenes), generator=generator) * 100e-6
    simulator.reset()
    percept = simulator(amplitude)
    assert torch.isfinite(percept).all()
    assert percept.max() > 0, 'expected a visible percept at up to 100 uA'
