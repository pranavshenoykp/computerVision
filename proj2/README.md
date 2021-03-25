# CS 6476 Project 2: Feature Matching

# Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses python3 anyways.
2. Download and extract the project starter code.
3. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj2_env_<OS>.yml`
4. This will create an environment named 'cs6476_proj2'. Activate it using the Windows command, `activate cs6476_proj2` or the MacOS / Linux command, `conda activate cs6476_proj2` or `source activate cs6476_proj2`
5. Install the project package, by running `pip install -e .` inside the repo folder. This might be unnecessary for every project, but is good practice when setting up a new `conda` environment that may have `pip` requirements.
6. Run the notebook using `jupyter notebook ./proj2_code/proj2.ipynb`
7. After implementing all functions, ensure that all sanity checks are passing by running `pytest proj2_unit_tests` inside the repo folder.
8. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`


# FAQ

## Installation

- There is no module named 'proj2_code' Getting "ModuleNotFoundError: No module named 'proj2_code' when running the setup in proj2 notebook." How can I fix this? Did you already install the modules with ‘pip install -e .’ inside the project root directory?


## Harris

- Why are my interest points all over the place? Are you sure you are sorting by confidence and only returning the top K most confident keypoints? What do your top 10 keypoints look like?

- I have an "index out of bound error"? Images are given in height x width order. Height corresponds to y values and width corresponds to x values, so you should index with y first and x second.

- Can we change the default value of num_points from 4500 to something smaller if it still passes the Notre Dame 100 points with above 80% accuracy? Or will this affect other tests the TAs are running? You're welcome to play around with the number of keypoints, but the runtime and accuracy unit tests need to pass (without modifying the unit test code) to get credit for those components.

- What do Ix and Iy refer to?  Ix, Iy are the image gradients in the x-direction, and y-direction. You can see how they are used in Equations 7.1, 7.8 of the Szeliski textbook (pages 208-212). We want you to use Sobel filters/operators to extract these gradients. (https://en.wikipedia.org/wiki/Sobel_operator)

- Should we compute Ixx, Iyy, and Ixy by differentiating Ix and Iy a second time? Are they "second order derivatives"? No, Ixx, Iyy, and Ixy are mixed derivatives, that are just the outer product of [Ix, Iy] with itself. $$\begin{bmatrix} I_x \\ I_y \end{bmatrix} \begin{bmatrix} I_x & I_y \end{bmatrix}$$

- What is `remove_border_vals()` for? How is it used, what are its inputs and what is expected to perform in here? Thee idea is that after you find pixels with high probability of being corners, we'll need to ignore those that lie near the borders of the image. This is because our keypoint "descriptor" requires a 16x16 window (image patch), centered at the keypoint's location. So after you figure out which pixels are suitable corners, please prune out the border ones.



## Parameters for Convolution

- What is the difference between applying convolution with a Gaussian filter and image filtering (cross-correlation) in Project 1? There is no difference, since a Gaussian filter is symmetric along all axes, so the two operations are equivalent.

- I'm trying to use the Gaussian filter, but its 2 dimensional and I keep getting an error that the weights need to be 3 dimensional? Yes, to initialize the weight you do need to reshape the filter into the correct shape based on the input to the layer. If you are ever curious what size a weight a "Conv" layer requires, you can initialize the appropriate torch.nn convolution operation, and then print out the weight needed, e.g., `self.myoperation = torch.nn.MyOperationName(); print(self.myoperation.weight.shape)` You can reshape the pre-existing values. Here's an example of how to reshape tensors. Here's an example of how to examine what size of tensor you need for your weight layer. https://johnwlambert.github.io/pytorch-tutorial/#reshaping, https://johnwlambert.github.io/pytorch-tutorial/#conv-weights

- What is the batch size? `num_image` represents the batch dimension (how many images you are sending through your network at once). For the purpose of this project, you can assume num_image=1. PyTorch allows sending hundreds/1000s etc of images through your network, applying the same operations to each image independently, which can make things quite fast.

- I get `RuntimeError: Expected object of scalar type Float but got scalar type Double for argument #2 'weight'.`? You may be inserting a weight tensor into your ConvLayer that is of the wrong data type (perhaps should be float, but you inserted double). You can read about available data types and how to cast a tensor to the correct data type in our PyTorch tutorial linked here. https://johnwlambert.github.io/pytorch-tutorial/#data-types


## SIFT

- Since the window is 16x16, the keypoint cannot be centered in the window. How should we handle that? Of the 4 potential choices for the center, please use the top-left (see project handout).

- What should we do if the 16x16 window falls off the edge of the image? We would be accumulating padded (completely hallucinated and erroneous) gradient values into the keypoint's descriptor, so we will throw away any keypoints that would meet this undesirable condition.


## Feature Matching

- NNDR? You can read more about it in Section 7.1 Keypoint matching of David Lowe's 2004 paper (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).  Page 425 of book. we want to get the matches only when it was similar to one unique point instead of being close with many points. For example, in a skyscraper with lots of windows, you will get many similar points. The ratio tests prevents matches between them. Using the SIFT descriptor matching, each window can be matched with any other window. Hence they should be avoided.

- Szeliski on SIFT feature clipping: According to Szeliski, "To reduce the effects of contrast or gain (additive variations are already removed by the gradient),the 128-D vector is normalized to unit length. To further make the descriptor robust to other photometric variations, values are clipped to 0.2 and the resulting vector is once again renormalized to unit length." If I don't do this, `test_get_sift_feature_descriptors` passes the sum check, but I get a tiny bit better accuracy if I implement this. Will I lose points if I fail the sum check due to this clip & normalize step? That would be very cool if you could run some experiments and discuss the improved results in your report.  Please submit the version without clipping in your code submission though (just comment out the clipping lines so we can check against our unit tests).

- Will our implementation of `compute_feature_distances` be graded? I ask this because I was thinking of computing (f1 - f2) ^ 2 rather than the distance. If I leave out the square root that should save some (likely tiny) amount of time, and minimizing distance squared is equivalent to minimizing distance. Then I can simply take the square root of the nearest neighbor ratio each time I calculate it (saving on square root operations). Is this okay, or must I return the actual distance in `compute_feature_distances`? This is a good observation, but we will be grading the `compute_feature_distances()` unit test. The square root should take very little time in Numpy, so I think optimization is better spent elsewhere.

- I get an error `IndexError: arrays used as indices must be of integer (or boolean) type"` Have you checked that you are returning arguments of the correct type? Can you put a breakpoint right before the line where it crashes, and check the sizes of your arrays x1, x2, and matches? If you have zero correct matches, it might be crashing.

- How to vectorize? I was able to get `compute_feature_distances` to run within 2-3 seconds using 1 for-loop, so that should be optimization enough if that's the problem. (hint: if the distance table is NxM, and each row has the distances of features1[row] to all of the the "M" features2 vectors, how can we compute a vector of all the differences between the features1[row] vector and all the features2 ones?)


## Runtime + Accuracy + Memory

- How long should it take? Around 30s for the entire pipeline on Notre Dame.

- High matches, high accuracy, but takes a really long time. Is there anything that can be done to significantly reduce my time? Some of the cases are taking 5-10 minutes? Break down which part of the code is taking the most time first. If there are for loops, perhaps a vectorized solution would be faster. Minimize any extra computations, etc. In general a solution going over the time limit will have a small deduction, so focus on getting everything else done first. Vectorize both of the methods in `part3_feature_matching.py`


- How do you check for time? Essentially a single run of the notebook (Harris, SIFT, matching for an image pair)

- How do I improve accuracy? Try using more corners. You should return arrays x,y,c from get_interest_points() that are sorted so that the first x-entry and the first y-entry correspond to the most confident keypoint (highest cornerness value), and descending from there. You can choose the top K=100 cornerness values, K=500, K=4500, etc (K representing num_points) to see how many you need to get good accuracy. How many keypoints are you using? Are you passing each of the unit tests? Are you passing the feature matching tests?
definitely recommend "profiling" your code -- seeing how long each subroutine takes. You can use
```
import time
start = time.time()
# your code here
end = time.time()
duration = end - start
print(f'My code took {duration} seconds to run')
```

- `get_sift_feature_descriptors()` is for sure taking a long time. Is there some way to vectorize that as well? Right now, I have two nested for loops. The outer loops through the list of (x,y) coordinates and the inner loops through the 16 neighbors of each coordinate. Remember that you can index into arrays in Numpy and Pytorch with arrays of indices.


## Unit Tests

- I get syntax errors when I run `pytest proj2_unit_tests`? Can you type `which python` and  `python --version` in your terminal? You might be using an older version of Python (for example, before version 3.6 you couldn't form strings with the f'mystring' syntax).


## Report

- Number of interest points for report -- Do we use 4500 interest points for all visualizations in the report for project 2?  No, use whatever performs best. One is okay, you can squeeze both there though.


## Other

- You can read more about how to go from one feature map size to another feature map size here under "Summary". (http://cs231n.github.io/convolutional-networks/) Specifically, a conv layer:

http://cs231n.github.io/convolutional-networks/
Accepts a volume of size W1×H1×D1
Requires four hyperparameters:
Number of filters K,
their spatial extent F,
the stride S,
the amount of zero padding P.
Produces a volume of size W2×H2×D2 where:
W2=(W1−F+2P)/S+1
H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
D2=K

