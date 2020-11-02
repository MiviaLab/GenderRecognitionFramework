from cv2 import cv2
import numpy as np
import os 

EXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_models")
modelFile = os.path.join(EXT_ROOT, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
configFile=os.path.join(EXT_ROOT, "deploy.prototxt")

class FaceDetector:
    net = None
    def __init__(self, min_confidence=0.5):
        print ("FaceDetector -> init")
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.min_confidence = min_confidence
        print ("FaceDetector -> init ok")
    
    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        frameHeight, frameWidth, channels = image.shape
        self.net.setInput(blob)
        detections = self.net.forward()
        faces_result=[]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                f = (x1,y1, x2-x1, y2-y1)
                if f[2]>1 and f[3]>1:
                    faces_result.append({
                        'roi': f,
                        'type': 'face',
                        'img': image[f[1]:f[1]+f[3], f[0]:f[0]+f[2]],
                        'confidence' : confidence
                    })
        return faces_result
    
    def __del__(self):
        print ("FaceDetector -> bye")


def top_left(f):
    return (f['roi'][0], f['roi'][1])
	
def bottom_right(f):
    return (f['roi'][0]+f['roi'][2], f['roi'][1]+f['roi'][3])

def enclosing_square(rect):
    def _to_wh(s,l,ss,ll, width_is_long):
        if width_is_long:
            return l,s,ll,ss
        else:
            return s,l,ss,ll
    def _to_long_short(rect):
        x,y,w,h = rect
        if w>h:
            l,s,ll,ss = x,y,w,h
            width_is_long = True
        else:
            s,l,ss,ll = x,y,w,h
            width_is_long = False
        return s,l,ss,ll,width_is_long

    s,l,ss,ll,width_is_long = _to_long_short(rect)

    hdiff = (ll - ss)//2
    s-=hdiff
    ss = ll

    return _to_wh(s,l,ss,ll,width_is_long)

def add_margin(roi, qty):
    return (
     roi[0]-qty,
     roi[1]-qty,
     roi[2]+2*qty,
     roi[3]+2*qty )

def cut(frame, roi):
    pA = ( int(roi[0]) , int(roi[1]) )
    pB = ( int(roi[0]+roi[2]), int(roi[1]+roi[3]) ) #pB will be an internal point
    W,H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0]>=0 else 0
    A1 = pA[1] if pA[1]>=0 else 0
    data = frame[ A1:pB[1], A0:pB[0] ]
    if pB[0] < W and pB[1] < H and pA[0]>=0 and pA[1]>=0:
        return data
    w,h = int(roi[2]), int(roi[3])
    img = np.zeros((h,w,frame.shape[2]), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0]<0 else 0
    offY = int(-roi[1]) if roi[1]<0 else 0
    np.copyto( img[ offY:offY+data.shape[0], offX:offX+data.shape[1] ], data )
    return img


def findRelevantFace(objs, W,H):
    mindistcenter = None
    minobj = None
    for o in objs:
        cx = o['roi'][0] + (o['roi'][2]/2)
        cy = o['roi'][1] + (o['roi'][3]/2)
        distcenter = (cx-(W/2))**2 + (cy-(H/2))**2
        if mindistcenter is None or distcenter < mindistcenter:
            mindistcenter = distcenter
            minobj = o
    return minobj