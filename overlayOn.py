
import cv2
import numpy as np
import os
import random
import imutils
import argparse
import image_augmentation as ia


def overlay(limg, simg, xOff, yOff):
    a1 = yOff+simg.shape[0]
    a2 = xOff+simg.shape[1]
    a5 = simg.shape[0]
    a6 = simg.shape[1]
    if a1 > limg.shape[0]:
        a5 = simg.shape[0] - (a1 - limg.shape[0])
        a1 = limg.shape[0]
    if a2 > limg.shape[1]:
        a6 = simg.shape[1] - (a2 - limg.shape[1])
        a2 = limg.shape[1]
    try:
        limg[yOff:a1, xOff:a2] = simg[:a5, :a6]
    except:
        pass
    return limg


parser = argparse.ArgumentParser()
# parser.add_argument('--mainPath', type=str, default='', help='main folder path')
parser.add_argument('--inPath', type=str, default='', help='input images folder path')
parser.add_argument('--outPath', type=str, default='', help='output images folder path')
parser.add_argument('--smallPath', type=str, default='', help='small images folder path')
parser.add_argument('--imgHeight', type=int, default=416, help='image height')
parser.add_argument('--imgWidth', type=int, default=416, help='image width')
parser.add_argument('--imgChannel', type=int, default=3, help='image depth')
args = parser.parse_args()

inpath = args.inpath
outpath = args.outPath

if not os.path.exists(outpath):
    try:
        os.makedirs(outpath)
    except:
        raise

# functions to use from image augmentation
funks = [ia.randomRotate, 
            ia.randomBrightness, 
            ia.randomContrast, 
            ia.randomSaturation, 
            ia.injectNoise, 
            ia.randomShearing, 
            ia.randomDropout, 
            ia.pooling,
            ia.skip]

bigWidth, bigHeight, _ = args.imgWidth, args.imgHeight, 3

# small images names
sms = ['crosswalk.png','parkingmain.jpg', 'giveWay.png', 'stopmain.png', 'straight.png', 'turnLeft.png', 'turnRight.png']

n = 0

for fold in ['1', '2', '3', '4', '5', '6', '7', '8']:  # different folders for different signs
    sm = cv2.imread(os.path.join(args.smallPath, str(sms[int(fold) - 1])))
    simg = cv2.resize(sm, (50, 50))
    for filename in os.listdir(os.path.join(args.inPath, fold)):  # +'/'
        if filename[-3:] == 'jpg':
            n += 1
            image = cv2.imread(os.path.join(inPath, fold, filename))
            newsmall = ia.scaling(simg)
            funk = np.random.choice(funks)
            print(funk)
            xoff = random.randint(0, bigWidth - newsmall.shape[1])
            yoff = random.randint(0, bigHeight - newsmall.shape[0])
            out = overlay(image, newsmall, xoff, yoff)
            if funk == ia.randomDropout:
                img = funk(out, newsmall.shape[1], newsmall.shape[0], xoff, yoff)
            else:
                img = funk(out)
            cv2.imwrite(os.path.join(outPath, str(n) + '.jpg'), img)
            print('wrote: ', str(n)+'.jpg')
        

