# Dolly zoom effect (Vertigo effect)

Dolly Zoom effect is used by filmmakers to create a sensation of vertigo, a “falling-away-from-oneself feeling”. This effect keeps the size of an object of interests constant in the image, while making the foreground and background objects appear larger or smaller by adjusting focal length and moving the camera.

The technique used here employs a simple cellphone camera without a zoom lens and tries to get the same results
by means of cropping and warping the images taken at multiple distances.

![](https://github.com/pranavshenoykp/computerVision/blob/main/DollyZoom/small_boat1.gif)

# Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses python3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command.
3. This will create an environment named 'dolly_zoom'. Activate it using the Windows command, `conda activate dolly_zoom`.
4. Run the notebook using `jupyter notebook ./dollyZoom.ipynb`
5. You run the script with your own imageset by saving the images in ./data/<folder>. Remember to sort the images in ascending order of distance.

## Credits and references

1. https://github.com/chetansastry/dolly-zoom
2. Robotics:Perception Assignment 1 by University of Pennsylvania on Coursera. (https://www.coursera.org/learn/robotics-perception)
3. https://stackoverflow.com/questions/55149171/how-to-get-roi-bounding-box-coordinates-with-mouse-clicks-instead-of-guess-che
