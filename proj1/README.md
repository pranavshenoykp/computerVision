# CS 6476 project 1: Convolution and Hybrid Images

# Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses python3 anyways.
2. Download and extract the project starter code.
3. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj1_env_<OS>.yml`
4. This will create an environment named 'cs6476_proj1'. Activate it using the Windows command, `activate cs6476_proj1` or the MacOS / Linux command, `conda activate cs6476_proj1` or `source activate cs6476_proj1`
5. Install the project package, by running `pip install -e .` inside the repo folder. This might be unnecessary for every project, but is good practice when setting up a new `conda` environment that may have `pip` requirements.
6. Run the notebook using `jupyter notebook ./proj1_code/proj1.ipynb`
7. After implementing all functions, ensure that all sanity checks are passing by running `pytest proj1_unit_tests` inside the repo folder.
8. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`

## FAQ

**Python**
- How do i start? First off, I would head to https://cs231n.github.io/python-numpy-tutorial/ to read a Python and Numpy tutorial. This should give you a good understanding of the Numpy basics you'll need for this assignment. Afterwards, you might find this [brief iPython notebook tutorial](https://github.com/cs231n/cs231n.github.io/blob/7a3f2a9c79f3599b4253f5ed871f4ba8bfe72c65/jupyter-notebook-tutorial.ipynb) useful.

**Gaussian Kernel**
- *Can I create a Gaussian kernel by drawing random samples from np.random.normal?* Sampling random values from a Gaussian isn't what we need here, since we have to enforce a very specific spatial layout. Indeed, if you were to draw a large numbers of values x∼N(0,I), e.g. from np.random.normal, their values would follow a Gaussian PDF. Rather, what we need you to do is generate a 2d grid of intensity values, where intensity values depend upon coordinates in the grid (following the Gaussian PDF). Values should tail off from the peak (which occurs at the mean coordinate value). If you insert the value of the exact Gaussian PDF at each coordinate of the kernel Gij, the values will not sum to one. however, with a constant scaling α to every Gij value, the kernel will sum to one. In other words, $$ \sum_{ij} \alpha G_{ij} = 1 $$.
- *What is the difference (meaning) of a 1D vs 2D Gaussian Kernel?*  The functions we define can take in one input variable (e.g. f(x)=x2), or it can take more than one input variable (e.g. f(x,y)=x2+y2). The first example is a 1D function and the second is a 2D function. Similarly, the Gaussian kernel can be defined in any d-dimensional space as the input. For the 1D case, we have a scalar mean and scalar variance for defining the Gaussian. Similar function can be defined for higher dimensions using mean vector and the covariance matrix. This is a useful reference: http://cs229.stanford.edu/section/gaussians.pdf
- What is "x" in the Gaussian PDF equation? x is the instantiation of the random variable (taking on a specific value), and p(x;μ,σ^2) is the density at that value. Thus, to populate a 1d Gaussian kernel, you will populate the filter values with the corresponding probability density. x would be a value in a discrete number line interval (a location on the interval), μ would be the average value in the discrete interval. To extend this to 2d, you'll have x∈R^2, instead of x∈R. x is a discrete index in a number range. μ, on the other hand, is the midpoint of that number range and is the mean. In the 1d case, "x" could be locations on a number line interval, e.g. [0,1,2,3,4], or [-2,-1,0,1,2]. Consider what the midpoint would be for either toy example. We'll then feed such values into the univariate Gaussian density function, and obtain scalar density values at each number line point.
- *What does the "exp" portion of the multivariate Gaussian function stand for?* exp(x) is e^x.

**Image Filtering**
- *Should i implement cross-correlation or convolution?* It won't matter, since we'll give you symmetric filters in the unit tests. Convolution (Equation 3.14 on page 118 of the book, page 142 of the book PDF) is equivalent to image filtering (cross-correlation) with a vertically and horizontally flipped filter (mirroring about the center anchor point).
- *I have this black border around the image and i'm not sure how to fix it?* There are a couple of things you can try to answer/verify, which will help you debug the code.
Why is the padding needed? What is the width of padding on each edge of the image? What is the index on the padded image that you need to run the filter on?
- *I've successfully implemented the my_conv2d_numpy() function but the runtime for the Jupyter notebook code snippet that runs it on the dog picture is around 8-10 minutes, which is over 5 minutes. Any tips on how to optimize?* Try to vectorize your code by replacing for loops with matrix operations in NumPy. You might find np.sum() and np.multiply() helpful as suggested in the project writeup to speed up your code.
- *My code doesnt work. What do I do?* Set a breakpoint in your code with import pdb; pdb.set_trace(), and step through it with "n", or "s" or "c"
usually debug a forward() call by importing pdb in that file, and then setting pdb.set_trace() inside the forward function. Then you can step through and check the sizes, one call at a time.
- *How would I know how many zeros to place around the entire boundary of the image?* It's based on how large your filter is. It might help to draw out some examples using different filter sizes. You can start to see just how many extra padded rows or columns you might need.
- *I'm confused about the image format in my_conv2d_numpy()*. You take each color channel separately and then apply the filter to it.

**Dataloader**
- *My dataloader isn't loading images properly. What's going on?* Note that the Pillow image library loads images in height,width,channel (HWC) order, whereas Pytorch requires CHW order. Note also the return types of each funtion in the function signature -- i.e. cutoff_frequency should be converted to a Tensor before sending it to the model.
- *Does "make_dataset" method have to be robust? When creating the list of paths to each image, can we hard-code the method to fit the files we already have in the folder for the project? Or are we expected to make a robust method that will work regardless of what is in the folder?* You should always avoid hard-coding. You can assume that we will always use the same prefix naming scheme to create pairs ("1a", "1b", "2a", "2b", ..., "#a", "#b").
-* Will there always be an underscore following the prefix number letter scheme? So can we always assume it will be "2a_", "2b_", etc.?* Yes
- How should I create the dataset? You will have to parse the filename to determine whether it is an “a” image or a “b” image and put it in a corresponding list.
I used regex to do this, keep in mind the filenames are ###(a or b)_nameofimage.bmp They should be ordered lists though (so the same index in both lists are corresponding pictures). Yes, any built-in Python libraries (e.g. ones that don't require installation via conda/pip/other methods) are allowed.

**Hybrid Images**
- *Where can I find the original Hybrid Images paper?* Here: http://olivalab.mit.edu/publications/OlivaTorralb_Hybrid_Siggraph06.pdf

**PyTorch**
- For my torch.nn.functional.conv2d function, I am having the issue of  "RuntimeError: Input type (torch.FloatTensor) and weight type (torch.DoubleTensor) should be the same."
To use torch.nn.functional.conv2d you need to define the input and the weight parameters before passing them to the function. Before you pass them to the function you should be able to make sure that the types match.
- *Batch -- Why is there a tensor of size b? * B denotes the batch size. The API should take care of multiple images if they are passed as input. The filter is same for all the images in a batch. Its very common in ML applications to apply the operations for a batch instead of manually looping, as it is more efficient and easy to write. Batch size is 1 by default throughout this project.
- *I have a type error. What do I do?* Check out https://discuss.pytorch.org/t/how-to-cast-a-tensor-to-another-type/2713, e.g. convert float to double.
-Why do we need padding? Zero padding means we create an image that is slightly larger than our original image by placing zeros around the edges. This way when applying the filter to our larger image, it outputs something that's the correct size
