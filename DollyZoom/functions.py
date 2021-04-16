import cv2
import numpy as np
import utils
from cv2 import ORB


def single2im(im: np.ndarray) -> np.ndarray:
    """
    Args:
        im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
        im: uint8 array of identical shape and in range [0,255]
    """
    im *= 255
    im = im.astype(np.uint8)
    return im


def Get_matches(pic_a: np.ndarray, pic_b: np.ndarray, n_feat: int) -> (np.ndarray, np.ndarray):
    """Get unreliable matching points between two images using SIFT.

    You do not need to modify anything in this function, although you can if
    you want to.

    Args:
        pic_a: a numpy array representing image 1.
        pic_b: a numpy array representing image 2.
        n_feat: an int representing number of matching points required.

    Returns:
        pts_a: a numpy array representing image 1 points.
        pts_b: a numpy array representing image 2 points.
    """
    pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)

    pic_a = single2im(pic_a)
    pic_b = single2im(pic_b)

    sift = cv2.xfeatures2d.SIFT_create()

    kp_a, desc_a = sift.detectAndCompute(pic_a, None)
    kp_b, desc_b = sift.detectAndCompute(pic_b, None)
    dm = cv2.BFMatcher(cv2.NORM_L2)
    matches = dm.knnMatch(desc_b, desc_a, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance / 1.2:
            good_matches.append(m)
    pts_a = []
    pts_b = []
    for m in good_matches[: int(n_feat)]:
        pts_a.append(kp_a[m.trainIdx].pt)
        pts_b.append(kp_b[m.queryIdx].pt)

    return np.asarray(pts_a), np.asarray(pts_b)


def Get_matches2(pic_a: np.ndarray, pic_b: np.ndarray, n_feat: int) -> (np.ndarray, np.ndarray):

    pic_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    pic_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)

    pic_a = single2im(pic_a)
    pic_b = single2im(pic_b)

    kp_a = None
    kp_b = None
    desc_a = None
    desc_b = None

    orb = cv2.ORB_create()
    kp_a, desc_a = orb.detectAndCompute(pic_a, None)
    kp_b, desc_b = orb.detectAndCompute(pic_b, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)

    matches = sorted(matches, key=lambda x: x.distance)[:n_feat]

    pts_a = []
    pts_b = []

    for m in matches:
        pts_a.append(kp_a[m.queryIdx].pt)
        pts_b.append(kp_b[m.trainIdx].pt)

    return np.asarray(pts_a), np.asarray(pts_b)


def Get_homography_matrix(pts_a, pts_b):
    src_pts = np.zeros([len(pts_a), 1, 2])
    dst_pts = np.zeros([len(pts_b), 1, 2])

    for i in range(len(pts_a)):
    	src_pts[i, 0, :] = pts_a[i,:]
    	dst_pts[i, 0, :] = pts_b[i,:]

    transform, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return transform

def Apply_transform(next_img, transform, img):
	return cv2.warpPerspective(next_img, transform, img.shape[1::-1])