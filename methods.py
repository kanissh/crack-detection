import cv2 as cv
import numpy as np


def adjust_contrast(image, alpha, beta):
    result_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result_image


def filter(image):
    result_image = cv.equalizeHist(image)
    # result_image = cv.GaussianBlur(result_image, (3, 3), 0)
    result_image = cv.bilateralFilter(result_image, 5, 75, 75)
    return result_image


def gamma_correction(image, gamma):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv.LUT(image, lookUpTable)
    return res


def invert_image(image):
    return ~image


def thresh_otsu(image):
    th, image_otsu = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
    return image_otsu


def draw_contours(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def remove_contours(image, contours, contour_area):
    mask = np.ones(image.shape, dtype=np.uint8) * 255

    for c in contours:
        if cv.contourArea(c) < contour_area:
            cv.drawContours(mask, [c], -1, 0, -1)

    return cv.bitwise_and(image, image, mask=mask)


def draw_contour_on_image(image, image_bw_marked):
    contours, hierarchy = cv.findContours(image_bw_marked, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cv.drawContours(image, contours, -1, (0, 0, 255), 2)
