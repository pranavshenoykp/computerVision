# CS 6476 Project 4: Depth Estimation using Stereo

## Setup
Note that the proj3 environment should work for this project! If you run into import module errors, try `pip install -e .` again, and if that still doesn’t work, you may have to create a fresh environment.

1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj4_env_<OS>.yml`
3. This should create an environment named 'proj4'. Activate it using the Windows command, `activate cs6476_proj4` or the MacOS / Linux command, `source activate cs6476_proj4`, `conda activate cs6476_proj4`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook ./proj4_code/part1_simple_stereo.ipynb`
6. Run the notebook uploading the `mc_cnn.ipynb` to Colab and upload your implemented solutions for `part2a_network.py` and `part2b_patch.py`.
6. Ensure that all sanity checks are passing by running `pytest` inside the "proj4_unit_tests/" folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>` and submit to Gradescope!
