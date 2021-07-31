import numpy as np
import cv2
import imutils
import random
import math


def horizontalFlip(arr):
    return np.flip(arr, 1)


def verticalFlip(arr):
    return np.flip(arr, 0)


def randomRotate(img):
    maximum = 30
    angle = random.randint(-abs(maximum), abs(maximum))
    # return imutils.rotate(img, angle)
    image_center = tuple(np.array(img.shape[1::-1]) // 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1]) #  , flags=cv2.INTER_LINEAR


# def correctRandomRotate(img):
#     big = int(math.sqrt((img.shape[0] ** 2) + (img.shape[1] ** 2)))
#     new = np.zeros((big, big, 3))
#     xStart = (big - img.shape[1])//2
#     yStart = (big - img.shape[0])//2
#     new[yStart:yStart+img.shape[0], xStart:xStart+img.shape[1], :] = img
#     maximum = 40
#     angle = random.randint(-abs(maximum), abs(maximum))
#     # return imutils.rotate(img, angle)
#     image_center = tuple(np.array(new.shape[1::-1]) // 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     ow = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
#     return (ow)
    # maximum = 90
    # angle = random.randint(-abs(maximum), abs(maximum))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # edged = cv2.Canny(gray, 20, 100)
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # if len(cnts) > 0:
    #     # grab the largest contour, then draw a mask for the pill
    #     c = max(cnts, key=cv2.contourArea)
    #     mask = np.zeros(gray.shape, dtype="uint8")
    #     cv2.drawContours(mask, [c], -1, 255, -1)
    #     # compute its bounding box of pill, then extract the ROI,
    #     # and apply the mask
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     imageROI = img[y:y + h, x:x + w]
    #     maskROI = mask[y:y + h, x:x + w]
    #     imageROI = cv2.bitwise_and(imageROI, imageROI, mask=maskROI)
    #     rotatedImage = imutils.rotate_bound(imageROI, angle)
    # return rotatedImage


def scaling(img):
    minWidth, maxWidth = 50, 600
    minHeight, maxHeight = 50, 600
    width = random.randint(minWidth, maxWidth)
    height = random.randint(minHeight, maxHeight)
    return cv2.resize(img, (width, height))


def invert(img):
    return 255-img


def randomBrightness(img):
    maximum = 150
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


def randomSaturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    maximum = 120
    value = random.randint(-abs(maximum), abs(maximum))
    perc = value // 255
    s = s * perc
    after = cv2.merge((h, s, v))
    return cv2.cvtColor(after, cv2.COLOR_HSV2BGR)


def randomHue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    maximum = 100
    value = random.randint(-abs(maximum), abs(maximum))
    perc = value // 180
    s = s * perc
    after = cv2.merge((h, s, v))
    return cv2.cvtColor(after, cv2.COLOR_HSV2BGR)


def injectNoise(img):
    mean = random.uniform(0.9, 1)
    standardDeviation = random.uniform(0.9, 1)
    gauss = np.random.normal(loc=mean, scale=standardDeviation, size=img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    return cv2.add(img, gauss)


def injectNoise2D(img):
    mean = random.uniform(0.9, 1)
    standardDeviation = random.uniform(0.9, 1)
    gauss = np.random.normal(loc=mean, scale=standardDeviation, size=img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
    return cv2.add(img, gauss)


def extract(img):
    nonzero = np.nonzero(img[:, :, 0])
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
    matrix = np.float32([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])
    new = cv2.warpPerspective(img, matrix, (int(cols * 2), int(rows * 2)))
    return extract(new)


def randomColorSpace(img):
    cslist = [cv2.COLOR_BGR2HSV,
              cv2.COLOR_BGR2HLS,
              cv2.COLOR_BGR2LAB,
              cv2.COLOR_BGR2XYZ,
              cv2.COLOR_BGR2YUV]
    return cv2.cvtColor(img, random.choice(cslist))


def randomDropout(img):
    c1, c2, c3 = cv2.split(img)
    windowW = random.randint(20, 70)
    windowH = random.randint(20, 70)
    windowX = random.randint(10, img.shape[1]-80)
    windowY = random.randint(10, img.shape[0]-80)
    window = np.ones((windowH, windowW)) * 128
    window = injectNoise2D(window)
    c1[windowY:windowY+windowH, windowX:windowX+windowW] = window
    c2[windowY:windowY+windowH, windowX:windowX+windowW] = window
    c3[windowY:windowY+windowH, windowX:windowX+windowW] = window
    return cv2.merge(c1, c2, c3)


def max_pool(img, factor: int):
    ds_img = np.full((img.shape[0] // factor, img.shape[1] // factor), -float('inf'), dtype=img.dtype)
    np.maximum.at(ds_img, (np.arange(img.shape[0])[:, None] // factor, np.arange(img.shape[1]) // factor), img)
    return ds_img


def pooling(img):
    kernelSize = 3
    bot = img.shape[0] % kernelSize
    rig = img.shape[1] % kernelSize
    padded = cv2.copyMakeBorder(img, 0, bot, 0, rig, cv2.BORDER_CONSTANT, value=[0,0,0])
    pooled1 = max_pool(padded[:, :, 0], kernelSize)
    pooled2 = max_pool(padded[:, :, 1], kernelSize)
    pooled3 = max_pool(padded[:, :, 2], kernelSize)
    extracted1 = extract(pooled1)
    extracted2 = extract(pooled2)
    extracted3 = extract(pooled3)
    extracted = cv2.merge((extracted1, extracted2, extracted3))
    return cv2.resize(extracted, img.shape)


def randomAffine(img):
    pass

