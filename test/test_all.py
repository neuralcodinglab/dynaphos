import cv2
import pytest
import torch
import numpy as np
import pandas as pd

from dynaphos.simulator import GaussianSimulator
from dynaphos.cortex_models import (
    get_visual_field_coordinates_from_cortex, remove_out_of_view,
    get_visual_field_coordinates_grid, get_mapping_from_cortex_to_visual_field,
    get_cortex_coordinates_default, get_cortex_coordinates_grid,
    get_visual_field_coordinates_from_cortex_full,
    get_visual_field_coordinates_probabilistically)
from dynaphos.utils import (to_tensor, to_numpy, get_data_kwargs, load_params,
                            load_coordinates_from_yaml, Map)


PARAMS_PATH = '../config/params.yaml'
MAPPING_MODELS = ['monopole', 'dipole', 'wedge-dipole']


@pytest.fixture
def params():
    return load_params(PARAMS_PATH)


@pytest.fixture
def rng(params):
    return np.random.default_rng(seed=params['run']['seed'])


@pytest.fixture
def simulator(params, rng):
    params['sampling']['sampling_method'] = 'center'
    return get_simulator(params, rng)


def get_simulator(params, rng):
    coordinates_cortex = load_coordinates_from_yaml(
        '../config/grid_coords_dipole_valid.yaml', n_coordinates=100, rng=rng)
    coordinates_cortex = Map(*coordinates_cortex)
    coordinates_visual_field = get_visual_field_coordinates_from_cortex_full(
        params['cortex_model'], coordinates_cortex, rng)
    return GaussianSimulator(params, coordinates_visual_field, rng)


@pytest.fixture
def stimulus(params, simulator, rng):
    shape = params['run']['resolution']
    image = rng.random(shape) * 255
    return simulator.sample_stimulus(image)


class TestInit:
    def test_init_probabilistically(self, params, rng):
        r_expected = [4.18866466, 1.41023323, 5.35768769, 3.33191491,
                      0.18516116, 7.47153754, 4.03653153, 4.3407978,
                      0.26523123, 1.47428929]
        phi_expected = [2.3297927, 5.82303616, 4.04552386, 5.16956368,
                        2.78605358, 1.427783, 3.48455899, 0.40097565,
                        5.20016002, 3.96886447]
        coordinates_cortex = get_visual_field_coordinates_probabilistically(
            params, len(r_expected), rng)
        r, phi = coordinates_cortex.polar
        assert np.isclose(r, r_expected).all()
        assert np.isclose(phi, phi_expected).all()


class TestCorticalModels:
    def test_init_covering_electrode_grid(self, params):
        x_expected = [0., 26.27987687, 52.55975374, 78.83963061, 0.,
                      26.27987687, 52.55975374, 78.83963061, 0., 26.27987687,
                      52.55975374, 78.83963061, 0., 26.27987687, 52.55975374,
                      78.83963061]
        y_expected = [-24.44135778, -24.44135778, -24.44135778, -24.44135778,
                      -8.14711926, -8.14711926, -8.14711926, -8.14711926,
                      8.14711926, 8.14711926, 8.14711926, 8.14711926,
                      24.44135778, 24.44135778, 24.44135778, 24.44135778]
        coordinates_cortex = get_cortex_coordinates_grid(
            params['cortex_model'], 4, 4)
        x, y = coordinates_cortex.cartesian
        assert np.isclose(x, x_expected).all()
        assert np.isclose(y, y_expected).all()

    def test_get_phosphene_map_from_electrodes(self, params, rng):
        p = params['cortex_model']
        r_expected = [17.3140795, 17.0009518]
        phi_expected = [-0.57599127, 0.52750565]
        coordinates_cortex = get_cortex_coordinates_grid(p, 4, 4)
        coordinates_visual_field = get_visual_field_coordinates_from_cortex(
            p, coordinates_cortex, rng)
        r, phi = coordinates_visual_field.polar
        assert np.isclose(r, r_expected).all()
        assert np.isclose(phi, phi_expected).all()

    def test_init_full_view(self):
        visual_field = get_visual_field_coordinates_grid()
        z_expected = np.load('data/z_full_view.npy')
        assert np.isclose(visual_field.complex, z_expected).all()

    @pytest.mark.parametrize('mapping_model', MAPPING_MODELS)
    def test_generate_cortical_map(self, params, mapping_model):
        p = params['cortex_model']
        p['model'] = mapping_model
        cortex_map = get_cortex_coordinates_default(p)
        x, y = cortex_map.cartesian
        coordinates = np.load(f'data/coordinates_{mapping_model}.npz')
        assert np.isclose(x, coordinates['x']).all()
        assert np.isclose(y, coordinates['y']).all()

    @pytest.mark.parametrize('mapping_model', MAPPING_MODELS)
    def test_generate_phosphene_map(self, params, mapping_model):
        p = params['cortex_model']
        p['model'] = mapping_model
        cortex_map = get_cortex_coordinates_default(p)
        cortex_to_visual_field = get_mapping_from_cortex_to_visual_field(p)
        z = cortex_to_visual_field(cortex_map.complex)
        z_expected = get_visual_field_coordinates_grid().complex
        assert np.isclose(z, z_expected).all()

    def test_filter_invalid_electrodes(self):
        coordinates_visual_field = get_visual_field_coordinates_grid()
        z_all = coordinates_visual_field.complex
        z = remove_out_of_view(z_all)
        valid_electrodes = np.load('data/valid_electrodes.npy')
        assert np.array_equal(z, z_all[valid_electrodes])


class TestSimulator:
    def test_generate_phosphene_maps(self, simulator):
        phosphene_maps = to_numpy(simulator.phosphene_maps)
        phosphene_maps_expected = np.load('data/phosphene_map.npy')
        assert np.isclose(phosphene_maps, phosphene_maps_expected).all()

    def test_update(self, simulator, stimulus):
        simulator.update(stimulus)
        sigma = to_numpy(simulator.sigma.get())
        trace = to_numpy(simulator.trace.get())
        sigma_expected = np.load('data/sigma.npy')
        trace_expected = np.load('data/trace.npy')
        assert np.isclose(sigma, sigma_expected).all()
        assert np.isclose(trace, trace_expected).all()

    def test_gaussian_activation(self, simulator, stimulus):
        simulator.update(stimulus)
        activation = to_numpy(simulator.gaussian_activation())
        activation_expected = np.load('data/activation.npy')
        assert np.isclose(activation, activation_expected).all()

    def test_call(self, simulator, stimulus):
        phosphenes = to_numpy(simulator(stimulus))
        phosphenes_expected = np.load('data/output.npy')
        assert np.isclose(phosphenes, phosphenes_expected).all()


class TestFunctional:
    def test_image(self, params, rng):
        params['thresholding']['use_threshold'] = False
        shape = params['run']['resolution']
        frame = cv2.imread('data/donders.png')
        frame = cv2.resize(frame, shape)
        frame = 255 - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = frame.clip(0, 1) * 255
        simulator = get_simulator(params, rng)
        stimulus = simulator.sample_stimulus(frame)
        phosphenes = to_numpy(simulator(stimulus))
        phosphenes_expected = np.load('data/phosphenes_donders.npy')
        assert np.isclose(phosphenes, phosphenes_expected).all()

    def test_brightness(self, params, rng):
        fps = 500
        stimulus_sequence = np.concatenate([np.ones(83), np.zeros(417)])
        data = pd.read_csv('data/Fernandez_2021_fig6A.csv')

        params['cortex_model']['dropout_rate'] = 0
        params['default_stim']['pw_default'] = 170e-6
        params['default_stim']['freq_default'] = 300
        params['run']['fps'] = fps
        params['thresholding']['use_threshold'] = False
        coordinates_cortex = Map(np.array([35.]), np.array([10.]))
        coordinates_visual_field = get_visual_field_coordinates_from_cortex(
            params['cortex_model'], coordinates_cortex, rng)
        simulator = GaussianSimulator(params, coordinates_visual_field, rng)

        n_phosphenes = len(coordinates_visual_field)
        data_kwargs = get_data_kwargs(params)
        electrodes = torch.ones(n_phosphenes, **data_kwargs)
        results = []
        for stim_condition, amplitude in enumerate(data.amplitude):
            simulator.reset()
            states = []
            for i, stim in enumerate(stimulus_sequence):
                simulator.update(electrodes * stim * amplitude)
                state = {key: val.item() for key, val in simulator.get_state().items()}  # state as numpy
                states.append(state)
                states[-1]['amplitude'] = stim * amplitude
                states[-1]['stim_condition'] = stim_condition
                states[-1]['time'] = i / fps

            results.append(pd.DataFrame(states))
        results = pd.concat(results, ignore_index=True)

        # Find the peaks in activation and brightness for each stim_condition
        brightness = []
        activation = []
        for i in results.stim_condition.unique():
            brightness.append(
                results.loc[results.stim_condition == i, 'brightness'].max())
            activation.append(
                results.loc[results.stim_condition == i, 'activation'].max())
        activation = np.array(activation)
        brightness = np.array(brightness)
        activation_expected = np.load('data/fernandez_activation_fit.npy')
        brightness_expected = np.load('data/fernandez_brightness_fit.npy')
        assert np.isclose(activation, activation_expected).all()
        assert np.isclose(brightness, brightness_expected).all()

    def test_dynamics(self, params, rng):
        stimulus_amplitude = 90e-6

        # Stimulation sequences for 1200 seconds
        fps = 256
        total_duration = 200  # seconds
        train_duration = 0.125  # seconds
        stim_moments = np.concatenate(
            [np.linspace(0, 200, 50, endpoint=False),  # Fast part
             np.linspace(200, 1200, 6, endpoint=True)])  # Slow part

        num_frames = int(fps * train_duration)
        stim_sequences = np.zeros(int(fps * total_duration))
        for t in stim_moments:
            idx = int(t * fps)
            stim_sequences[idx:idx + num_frames] = stimulus_amplitude

        params['cortex_model']['dropout_rate'] = 0
        params['default_stim']['pw_default'] = 100e-6
        params['default_stim']['freq_default'] = 200
        params['run']['fps'] = fps
        params['thresholding']['use_threshold'] = False
        coordinates_cortex = Map(np.array([35.]), np.array([10.]))
        coordinates_visual_field = get_visual_field_coordinates_from_cortex(
            params['cortex_model'], coordinates_cortex, rng)
        simulator = GaussianSimulator(params, coordinates_visual_field, rng)

        data_kwargs = get_data_kwargs(params)
        states = []
        for i, stimulus in enumerate(to_tensor(stim_sequences, **data_kwargs)):
            simulator.update(stimulus)
            state = {key: val.item() for key, val in simulator.get_state().items()}  # state as numpy
            states.append(state)

        results = pd.DataFrame(states)
        results['stimulation'] = stim_sequences
        results['fps'] = fps
        results['time'] = results.index.copy() / fps
        results_expected = pd.read_pickle('data/results_dynamics.pkl')
        assert np.isclose(results.to_numpy(),
                          results_expected.to_numpy()).all()
