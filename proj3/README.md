# CS 6476 project 3: [Camera Projection Matrix and Fundamental Matrix Estimation with RANSAC](https://www.cc.gatech.edu/~hays/compvision/proj3/proj3.pdf)


# Setup
- Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses python3 anyway.
- Create a conda environment, using the appropriate command. On Windows, open the installed "Conda prompt" to run this command. On MacOS and Linux, you can just use a terminal window to run the command. Modify the command based on your OS ('linux', 'mac', or 'win'): `conda env create -f proj3_env_<OS>.yml`
- This should create an environment named `cs6476_proj3`. Activate it using the following Windows command: `activate cs6476_proj3` or the following MacOS / Linux command: `conda activate cs6476_proj3` or `source activate cs6476_proj3`.
- Install the project package, by running `pip install -e .` inside the repo folder. This might be unnecessary for every project, but is good practice when setting up a new `conda` environment that may have `pip` requirements.
- Run the notebook using `jupyter notebook ./proj3_code/proj3.ipynb`
- After implementing all functions, ensure that all sanity checks are passing by running `pytest proj3_unit_tests` inside the repo folder.
- Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`