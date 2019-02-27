
import cv2


def zoom(frame=None,mirror=False,cam=None,scale=None):

    ret_val, frame = cam.read()
    if mirror:
        frame = cv2.flip(frame, 1)
    #get the webcam size
    height, width, channels = frame.shape
    #prepare the crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(scale*height/100),int(scale*width/100)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))
    small_frame = cv2.resize(resized_cropped, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    return rgb_small_frame