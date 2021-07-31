
import cv2
import numpy as np
import os
import random


def overlay(limg, simg, xOff, yOff):
    new = np.copy(limg)
    new[yOff:yOff+simg.shape[0], xOff:xOff+simg.shape[1]] = simg
    return new


def resize(inPath, res, outPath):
    for fileName in os.listdir(inPath):
        if fileName[-3:] == 'jpg':
            org = cv2.imread(inPath+'\\'+fileName)
            out = cv2.resize(org, (res[0], res[1]))
            cv2.imwrite(outPath+'\\'+fileName, out)


def do(inPath, simg, outPath):
    for fileName in os.listdir(inPath):
        if fileName[-3:] == 'jpg':
            org = cv2.imread(inPath + '\\' + fileName)
            bigHeight, bigWidth, _ = org.shape
            try:
                xoff = random.randint(0, bigWidth-simg.shape[1])
                yoff = random.randint(0, bigHeight-simg.shape[0])
                out = overlay(org, simg, xoff, yoff)
                cv2.imwrite(outPath+'\\'+fileName, out)
            except Exception as e:
                print(e)


s = cv2.imread('c1.jpg')
sm = cv2.resize(s, (int(s.shape[1]*0.4), int(s.shape[0]*0.4)))
inF = 'C:\\Users\\mil\\Desktop\\imgCls\\data\\new\\monitor'
outF = 'C:\\Users\\mil\\Desktop\\imgCls\\data\\new\\res'
reso = (416, 416)
do(inF, sm, outF)



