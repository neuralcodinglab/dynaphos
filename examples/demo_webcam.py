import sys
import time

import cv2
import numpy as np

from dynaphos.image_processing import sobel_processor, canny_processor
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import load_params, load_coordinates_from_yaml, Map
from dynaphos.cortex_models import \
    get_visual_field_coordinates_from_cortex_full


def main(params: dict, in_video: int):
    params['thresholding']['use_threshold'] = False
    coordinates_cortex = load_coordinates_from_yaml(
        '../config/grid_coords_dipole_valid.yaml', n_coordinates=100)
    coordinates_cortex = Map(*coordinates_cortex)
    coordinates_visual_field = get_visual_field_coordinates_from_cortex_full(
        params['cortex_model'], coordinates_cortex)
    simulator = GaussianSimulator(params, coordinates_visual_field)
    resolution = params['run']['resolution']
    fps = params['run']['fps']

    prev = 0
    cap = cv2.VideoCapture(in_video)
    ret, frame = cap.read()
    while ret:

        # Capture the video frame by frame
        ret, frame = cap.read()

        time_elapsed = time.time() - prev
        if time_elapsed > 1 / fps:
            prev = time.time()

            # Create Canny edge detection mask
            frame = cv2.resize(frame, resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

            method = params['sampling']['filter']
            if method == 'sobel':
                processed_img = sobel_processor(frame)
            elif method == 'canny':
                threshold = params['sampling']['T_high']
                processed_img = canny_processor(frame, threshold // 2,
                                                threshold)
            elif method == 'none':
                processed_img = frame
            else:
                raise ValueError(f"{method} is not a valid filter keyword.")

            # Generate phosphenes
            stim_pattern = simulator.sample_stimulus(processed_img)
            phosphenes = simulator(stim_pattern)
            phosphenes = phosphenes.cpu().numpy() * 255

            # Concatenate results
            cat = np.concatenate([frame, processed_img, phosphenes],
                                 axis=1).astype('uint8')

            # Display the resulting frame
            cv2.imshow('Simulator', cat)

        # the 'q' button is set as the quit button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _params = load_params('../config/params.yaml')
    _in_video = 0  # use 0 for webcam, or string with video path
    main(_params, _in_video)
    sys.exit()
