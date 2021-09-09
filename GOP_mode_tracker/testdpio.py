import numpy as np
import cv2
import time
from deep_sort import nn_matching
from deep_sort.detection import Detection
#from deep_sort.trackergop import Tracker
from deep_sort.trackerkalmangop import Tracker
#from deep_sort import generate_detectionsprofiler as gdet

from deep_sort import generate_detectionsORG as gdet

def timelinecal(totaltim,tim):
    for i in tim:
        totaltimp["traceEvents"].append(tim[i])
class enc(object):
    def __init__(self,model):
        self.encoders = gdet.create_box_encoder(model, batch_size=1)
    def run(self,bbox,image):
        print("encode",np.shape(image),bbox)
        features,times = self.encoders(image,bbox)
        print("feature in there",np.array(features))
        return np.array(features) ,times 
class tracker(object):
    def __init__(self,model,max_cosine_distance,nn_budget):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
       
    def reset(self):
        self.tracker.reset()
        
    def startrack1(self,bboxs,image,scores,tag,features):
        
        st1 = time.time()
        if len(features)>=1:
            detection = [Detection(bbox,score,feature) for bbox,score,feature in zip(bboxs,scores,features)]
        else:
            detection = []
        self.tracker.predict(tag)

        st3 = time.time()
        if tag:

            self.tracker.update(detection)

        st4 = time.time() 
        return self.tracker.tracks,st3-st1


        
        

