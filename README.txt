
# creating synthetic images for ObjectDetection task
(used for yolo object detection)


* image_augmentation : differnet functions to have on images

* getFrames : to get background images from video frames to overlay on
   --src : source video file
   --n : number of frames needed
   --out : output directory to save frames

* removeDuplicate : to remove duplicate photos, in case there are any
   --path : photos path

* overlayOn : main script
   --inPath : input images path
   --outPath : output path for images
   --smallPath : path to small images


