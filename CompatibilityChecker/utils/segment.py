import cv2
import numpy as np
from skimage import measure
from scipy import ndimage

def segment(img: np.ndarray) -> None:
    """
    Segment an image by thresholding, morphological operations, flood fill, and 
    retaining the largest contour. Saves the result as 'mask.png'.

    Args:
        img (np.ndarray): Input image in RGB format.

    Returns:
        None
    """
    # Thresholding
    img_gray = np.min(img, axis=2)
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 115, 8
    )
    img_thresh = cv2.bitwise_not(img_thresh)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morph = cv2.morphologyEx(
        img_thresh, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    img_morph = cv2.morphologyEx(
        img_morph, cv2.MORPH_DILATE, kernel, iterations=2
    )

    # Flood fill
    mask = np.zeros([img_morph.shape[0] + 2, img_morph.shape[1] + 2], np.uint8)
    mask[1:-1, 1:-1] = img_morph.copy()

    im_floodfill = np.full(img_morph.shape, 255, np.uint8)

    h, w = img_morph.shape[:2]
    for x in range(0, w - 1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w - 1, y), 0)

    # Ensure edges are not considered in contour finding
    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    # Retain largest contour
    mask2 = cv2.bitwise_not(im_floodfill)
    mask = None
    largest_contour_size = 0

    contours = measure.find_contours(mask2, 0.0)
    for contour in contours:
        contour_img = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(contour_img, [np.int32(contour)], 1)
        contour_size = len(np.where(contour_img == 1)[0])
        if contour_size > largest_contour_size:
            mask = contour_img
            largest_contour_size = contour_size

    if mask is None:
        raise ValueError('No contours found within image')

    mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = 255 * mask.astype(np.uint8)
    cv2.imwrite('characterfiles/mask.png', mask.T)
    print("--Masked image saved.")
