

import numpy as np
import cv2
import imutils
import random
import math


def horizontalFlip(arr):
    return np.flip(arr, 1)


def verticalFlip(arr):
    return np.flip(arr, 0)


def manualCrop():
    pass


def randomPadding():
    pass


def randomRotate(img):
    maximum = 40
    angle = random.randint(-abs(maximum), abs(maximum))
    # return imutils.rotate(img, angle)
    image_center = tuple(np.array(img.shape[1::-1]) // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def correctRandomRotate(img):
    big = int(math.sqrt((img.shape[0] ** 2) + (img.shape[1] ** 2)))
    new = np.zeros((big, big, 3))
    xStart = (big - img.shape[1])//2
    yStart = (big - img.shape[0])//2
    new[yStart:yStart+img.shape[0], xStart:xStart+img.shape[1], :] = img
    maximum = 40
    angle = random.randint(-abs(maximum), abs(maximum))
    # return imutils.rotate(img, angle)
    image_center = tuple(np.array(new.shape[1::-1]) // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    ow = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return (ow)

    maximum = 90
    angle = random.randint(-abs(maximum), abs(maximum))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 20, 100)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        # grab the largest contour, then draw a mask for the pill
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # compute its bounding box of pill, then extract the ROI,
        # and apply the mask
        (x, y, w, h) = cv2.boundingRect(c)
        imageROI = img[y:y + h, x:x + w]
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
        rotatedImage = imutils.rotate_bound(imageROI, angle)
    return rotatedImage


def scaling(img):
    minWidth, maxWidth = 50, 1280
    minHeight, maxHeight = 50, 720
    width = random.randint(minWidth, maxWidth)
    height = random.randint(minHeight, maxHeight)
    return cv2.resize(img, (width, height))


def invert(img):
    return 255-img


def randomBrightness(img):
    maximum = 100
    value = random.randint(-abs(maximum), abs(maximum))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    after = cv2.merge((h, s, v))
    return cv2.cvtColor(after, cv2.COLOR_HSV2BGR)


def randomContrast(img):
    alpha = round(random.uniform(0.5, 3), 1)
    return cv2.convertScaleAbs(img, alpha=alpha)


def randomSaturation():
    pass


def randomHue():
    pass


def Grayscale():
    pass


def injectNoise(img):
    mean = random.uniform(0.9, 1)
    standardDeviation = random.uniform(0.9, 1)
    gauss = np.random.normal(loc=mean, scale=standardDeviation, size=img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    return cv2.add(img, gauss)


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nonzero = np.nonzero(gray)
    maxWidth = max(nonzero[1])
    print(maxWidth)
    maxHeight = max(nonzero[0])
    print(maxHeight)
    return img[0:maxHeight, 0:maxWidth, :]


def randomShearing(img):
    maximum = 0.6
    shx = round(random.uniform(0, abs(maximum)), 1)
    shy = round(random.uniform(0, abs(maximum)), 1)
    rows, cols, dep = img.shape
    matrix = np.float32([[1, shx, 0],
                    [shy, 1, 0],
                    [0, 0, 1]])
    new = cv2.warpPerspective(img, matrix, (int(cols * 2), int(rows * 2)))
    return extract(new)


def cropAndPlace():  # import several off images for background
    pass


def randomColorSpace(img):
    cslist = [cv2.COLOR_BGR2HSV,
              cv2.COLOR_BGR2HLS,
              cv2.COLOR_BGR2LAB,
              cv2.COLOR_BGR2XYZ,
              cv2.COLOR_BGR2YUV]
    return cv2.cvtColor(img, random.choice(cslist))


'''orig = cv2.imread('name.jpg')
while True:
    rot3 = scaling(orig)
    cv2.imshow('rot3', rot3)
    cv2.waitKey()

cv2.destroyAllWindows()
'''
orig = cv2.imread('name.jpg')
newnew = correctRandomRotate(orig)
cv2.imshow('fuckfuckfuck', newnew)
cv2.waitKey()
cv2.destroyAllWindows()
