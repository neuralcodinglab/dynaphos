# Example videos

### Semantic segmentation example

![example_video.mp4](example_video.mp4)

Left: video and the semantic segmentation labels (from the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/)).
Middle: preprocessed image that was used to sample the electrode activations.
Right: output of the dynaphos phosphene simulator.

### Example videos gaze-contingent processing 

![example_video_gaze_contingent_object_grabbing.mp4](example_video_gaze_contingent_object_grabbing.mp4)

![example_video_gaze_contingent_walking.mp4](example_video_gaze_contingent_walking.mp4)

Left: input video. The red circle indicates the gaze-direction of the wearer of the camera.
Middle: the input video after processing with sobel edge detection (which was used to sample the electrode activations). The electrode activations are created by sampling the gaze-contingent patches (indicated by the circle). 
Right: output of the dynaphos phosphene simulator. The simulated phosphes are rendered contingent with the gaze direction.

**Acknowledgement:** Videos and gaze data are obtained by and used here with permission from Ashkan Nejad and Eva Postuma, Laboratory of Experimental Ophthalmology, University Medical Center Groningen.
