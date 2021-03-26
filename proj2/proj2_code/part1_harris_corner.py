#!/usr/bin/python3

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from typing import Tuple


SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use convolution with Sobel filters to compute the image gradient at each
    pixel.

    Args:
        image_bw: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    image_tensor = np.zeros([1,1,image_bw.shape[0],image_bw.shape[1]])
    image_tensor[0,0,:,:] = image_bw
    image_tensor = torch.Tensor(image_tensor)

    sobel_x_tensor = np.zeros([1,1,SOBEL_X_KERNEL.shape[0],SOBEL_X_KERNEL.shape[1]])
    sobel_x_tensor[0,0,:,:] = SOBEL_X_KERNEL
    sobel_x_tensor = torch.Tensor(sobel_x_tensor)

    sobel_y_tensor = np.zeros([1,1,SOBEL_Y_KERNEL.shape[0],SOBEL_Y_KERNEL.shape[1]])
    sobel_y_tensor[0,0,:,:] = SOBEL_Y_KERNEL
    sobel_y_tensor = torch.Tensor(sobel_y_tensor)

    Ix_tensor = F.conv2d(image_tensor, sobel_x_tensor, padding = 1, groups=1)
    Iy_tensor = F.conv2d(image_tensor, sobel_y_tensor, padding = 1, groups=1)

    Ix = Ix_tensor.numpy()[0,0,:,:]
    Iy = Iy_tensor.numpy()[0,0,:,:]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return Ix, Iy

def create_Gaussian_kernel_1D(ksize: int, sigma: float) -> np.ndarray:
    """Create a 1D Gaussian kernel using the specified filter size and standard deviation.
    
    The kernel should have:
    - shape (k,1)
    - mean = floor (ksize / 2)
    - values that sum to 1
    
    Args:
        ksize: length of kernel
        sigma: standard deviation of Gaussian distribution
    
    Returns:
        kernel: 1d column vector of shape (k,1)
    
    HINT:
    - You can evaluate the univariate Gaussian probability density function (pdf) at each
      of the 1d values on the kernel (think of a number line, with a peak at the center).
    - The goal is to discretize a 1d continuous distribution onto a vector.
    """
    kernel = np.ndarray(shape = (ksize,1), dtype=float)
    mean = np.floor(ksize/2)
    for x in range(ksize):
        kernel[x,0] = 1/(np.sqrt(2*np.pi) * sigma) * np.exp(-1/(2*sigma*sigma)*(x - mean)**2)
    
    kernel = kernel / np.sum(kernel)

    return kernel


def create_Gaussian_kernel_2D(ksize: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian kernel using the specified filter size, standard
    deviation and cutoff frequency.

    The kernel should have:
    - shape (k, k) where k = sigma * 4 + 1
    - mean = floor(k / 2)
    - standard deviation = sigma
    - values that sum to 1

    Args:
        sigma: an int controlling how much low frequency to leave in
        the image.
    Returns:
        kernel: numpy nd-array of shape (k, k)

    HINT:
    - You can use create_Gaussian_kernel_1D() to complete this in one line of code.
    - The 2D Gaussian kernel here can be calculated as the outer product of two
      1D vectors. In other words, as the outer product of two vectors, each 
      with values populated from evaluating the 1D Gaussian PDF at each 1d coordinate.
    - Alternatively, you can evaluate the multivariate Gaussian probability 
      density function (pdf) at each of the 2d values on the kernel's grid.
    - The goal is to discretize a 2d continuous distribution onto a matrix.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    kernel = np.dot(create_Gaussian_kernel_1D(ksize, sigma), create_Gaussian_kernel_1D(ksize, sigma).T)

    ### END OF STUDENT CODE ####
    ############################

    return kernel


def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel

    Args:
        ksize: dimension of square kernel
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel

    You should be able to reuse your project 1 Code here.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    kernel_numpy = create_Gaussian_kernel_2D(int(ksize), float(sigma))
    kernel = np.zeros([1,1,kernel_numpy.shape[0], kernel_numpy.shape[1]])
    kernel[0,0,:,:] = kernel_numpy
    kernel = torch.Tensor(kernel)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel


def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, the mixed derivatives,
    then the second moments (sx2, sxsy, sy2) at each pixel, using convolution
    with a Gaussian filter.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of Gaussian filter

    Returns:
        sx2: array of shape (M,N) containing the second moment in x direction
        sy2: array of shape (M,N) containing the second moment in y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the
            y direction
    """

    sx2, sy2, sxsy = None, None, None
    ###########################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                     #
    ###########################################################################

    Ix, Iy = compute_image_gradients(image_bw)
    Ixx_numpy = Ix * Ix
    Iyy_numpy = Iy * Iy
    Ixy_numpy = Ix * Iy

    Ixx = np.zeros([1,1,Ixx_numpy.shape[0], Ixx_numpy.shape[1]])
    Ixx[0,0,:,:] = Ixx_numpy
    Ixx = torch.Tensor(Ixx)

    Iyy = np.zeros([1,1,Iyy_numpy.shape[0], Iyy_numpy.shape[1]])
    Iyy[0,0,:,:] = Iyy_numpy
    Iyy = torch.Tensor(Iyy)

    Ixy = np.zeros([1,1,Ixy_numpy.shape[0], Ixy_numpy.shape[1]])
    Ixy[0,0,:,:] = Ixy_numpy
    Ixy = torch.Tensor(Ixy)

    gaussian_kernel = get_gaussian_kernel_2D_pytorch(int(ksize), float(sigma))

    padding = int((gaussian_kernel.shape[3] - 1)/2)

    sx2_tensor = F.conv2d(Ixx, gaussian_kernel, padding = padding, groups=1)
    sy2_tensor = F.conv2d(Iyy, gaussian_kernel, padding = padding, groups=1)
    sxsy_tensor = F.conv2d(Ixy, gaussian_kernel, padding = padding, groups=1)

    sx2 = sx2_tensor.numpy()[0,0,:,:]
    sy2 = sy2_tensor.numpy()[0,0,:,:]
    sxsy = sxsy_tensor.numpy()[0,0,:,:]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
        http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.
    You may find the Pytorch function nn.Conv2d() helpful here.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
            ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score

    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
    R = ((sx2 * sy2) - (sxsy * sxsy)) - alpha * (sx2 + sy2)**2 

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return R


def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d
            score/response map
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    R_tensor = np.zeros([1,1,R.shape[0], R.shape[1]])
    R_tensor[0,0,:,:] = R
    R_tensor = torch.Tensor(R_tensor)

    padding = int((ksize- 1)/2)

    maxpooled_R = F.max_pool2d(R_tensor, ksize, stride = 1, padding = padding)

    maxpooled_R = maxpooled_R.numpy()[0,0,:,:]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return maxpooled_R


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get top k interest points that are local maxima over (ksize,ksize)
    neighborhood.

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.
j
    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator

    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    median = np.nanmedian(R)
    R[R < median] = 0
    maxpool = maxpool_numpy(R, ksize)

    binary_R = np.copy(R)

    binary_R[binary_R != maxpool] = 0


    coordinates = np.where(binary_R > 0)
    confidences = binary_R[coordinates]

    coordinates = np.array(coordinates)

    idx = np.argsort(confidences)

    confidences = confidences[idx][::-1][:int(k)]
    coordinates = coordinates.T[idx][::-1][:int(k)]

    x = coordinates[:,0]
    y = coordinates[:,1]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, confidences


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,) representing x coord of interest points
        y: array of shape (k,) representing y coord of interest points
        c: array of shape (k,) representing confidences of interest points

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    h,w = img.shape

    h_min = 7
    h_max = h - 9

    w_min = 7
    w_max = w - 9

    x_mask = np.logical_and(x > h_min, x < h_max)
    y_mask = np.logical_and(y > w_min, y < w_max)
    mask = np.logical_and(x_mask, y_mask)

    x = x[mask]
    y = y[mask]
    c = c[mask]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the Harris Corner detector. You will find
    compute_harris_response_map(), nms_maxpool_pytorch(), and
    remove_border_vals() useful. Make sure to sort the interest points in
    order of confidence!

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: maximum number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        c: array of dim (p,) containing the strength(confidence) of each
            interest point where p <= k.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    R = compute_harris_response_map(image_bw)
    x, y, c = nms_maxpool_pytorch(R, k, ksize = 7)
    x, y, c = remove_border_vals(image_bw, x, y, c)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c
