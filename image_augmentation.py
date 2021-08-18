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


def scaling(img):
    minWidth, maxWidth = 40, 100
    minHeight, maxHeight = 40, 100
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
    s *= abs(perc)
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
    return cv2.add(np.asarray(img, np.uint8), gauss)


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


def randomDropout(img, maxW, maxH, x1, y1):
    x2 = x1 + maxW
    y2 = y1 + maxH
    windowW = random.randint(20, maxW)
    windowH = random.randint(20, maxH)
    windowX = random.randint(x1-10, x2+10)
    windowY = random.randint(y1-10, y2+10)
    window = np.ones((windowH, windowW, 3)) * 128
    window = injectNoise(window)
    endY = windowY + windowH
    endX = windowX + windowW
    if windowY+windowH > 415:
        endY = 415
    if windowX+windowW > 415:
        endX = 415
    try:    
        img[windowY:endY, windowX:endX, :] = window[:415-windowY, :415-windowX, :]
    except:
        pass
    return img


def max_pool(img, G):
    out = img.copy()
    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)
    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G*y:G*(y+1), G*x:G*(x+1), c] = np.max(out[G*y:G*(y+1), G*x:G*(x+1), c])
    return out


def pooling(img):
    kernelSize = 3
    bot = img.shape[0] % kernelSize + 1
    rig = img.shape[1] % kernelSize + 1
    padded = cv2.copyMakeBorder(img, 0, bot, 0, rig, cv2.BORDER_CONSTANT, value=[0,0,0])
    extracted = max_pool(padded, kernelSize)
    return cv2.resize(extracted, (img.shape[0], img.shape[1]))


def randomAffine(img):
    pass


def skip(img):
    return img