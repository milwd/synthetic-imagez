import cv2
import os
import argparse


'''
get = False

for filename in os.listdir():
    get = not get
    if get:
        os.remove(filename)
'''


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, help='Source video file')
parser.add_argument('--n', type=int, default=100, help='Number of frames')
parser.add_argument('--out', type=str, default=str(os.getcwd()), help='output directory')
args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(), args.out)):
    try:
        os.makedirs(os.path.join(os.getcwd(), args.out))
    except:
        raise


cap = cv2.VideoCapture(str(args.src))

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = int(cap.get(cv2.CAP_PROP_FPS))

every = frames // args.n
f = 0
while True:
    f += 1
    ret, frm = cap.read()
    frame = cv2.resize(frm, (416, 416))
    if f % every == 0:
        print(os.path.join(os.getcwd(), args.out ,str(f)[:6] + '.jpg'))
        cv2.imwrite(os.path.join(os.getcwd(), args.out ,str(f)[:6] + '.jpg'), frame)
        print('wrote frame: ', str(f))

cv2.destroyAllWindows()



