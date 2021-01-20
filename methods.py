import cv2 as cv
import numpy as np

def adjust_contrast(image, alpha, beta):
    # Image scaling
    result_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result_image


def filter(image):
    # Histogram equalization
    result_image = cv.equalizeHist(image)

    # Bilateral filtering
    result_image = cv.bilateralFilter(result_image, 5, 75, 75)
    return result_image


def gamma_correction(image, gamma):
    # Gamma correction
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv.LUT(image, lookUpTable)
    return res


def invert_image(image):
    # Image inversion
    return ~image


def thresh_otsu(image):
    # Otsu thresholding
    th, image_otsu = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    return image_otsu


def draw_contours(image):
    # Draw contours
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return contours


def remove_contours(image, contours, contour_area):
    # Remove single-point noises from the image
    mask = np.ones(image.shape, dtype=np.uint8) * 255

    for c in contours:
        if cv.contourArea(c) < contour_area:
            cv.drawContours(mask, [c], -1, 0, -1)

    image_result = cv.bitwise_and(image, image, mask=mask)
    return image_result


def draw_contour_on_image(image, image_bw_marked):
    # Mark crack in the image
    contours, hierarchy = cv.findContours(image_bw_marked, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image_result = cv.drawContours(image, contours, -1, (0, 0, 255), 2)

    return image_result
