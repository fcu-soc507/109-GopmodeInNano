"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO
from testdpio import tracker

from deep_sort import generate_detectionsTRT as gdet
WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-t', '--tracker_model', type=str,default = '0000',
        help='loading reid model')
    
    parser.add_argument(
        '-p', '--pframeset', type = str,default = 0000,
        help='gop setting enter "xxyy",xx is iframe number,yy is pframe number')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis,trackers,pframe,encode):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    #outputpath = "./pframe_output/o"+pframe+"/"
    full_scrn = False
    GETpath = "./V4GOPmoontestnew/"
    try:
        os.makedirs(GETpath+"p"+str(pframe))
    except:
        pass
    try:
        os.makedirs(GETpath+"n"+str(pframe))
    except:
        pass
    if (pframe==0):
        try:    
            os.makedirs(GETpath+"p100")
        except:
            pass
    cont = 0
    tim ,ct,ft,pt,ut,dt,det,mt= 0,0,0,0,0,0,0,0
    fps = 0.0
    goptag = 0
    gopcont = 0
    gopicont = 0
    totalcont = 0
    dcont = 0
    subcont = 0
    maxgop = pframe #MAX GOP frame
    #out = cv2.VideoWriter("./testmapAGILEV3Lbbb2turnnewmotion0101"+str(pframe)+"p.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
    out = cv2.VideoWriter(GETpath+str(pframe)+"p.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
    #path = os.listdir("./input")
    #path.sort(key=lambda x:int(x[2:-4]))
    #for imgname in path:
    #    cam = cv2.VideoCapture("./input/"+imgname)
    print("pframe",pframe)
    while True:
        cont +=1
        totalcont += 1
        outname = str(cont).rjust(5,'0')+".txt"
        #if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #    break
        try:
            img = cam.read() 
            img = cv2.resize(img,(640,480))  
        except:
            print("pframe",fps0)
            break
        tic = time.time()
        #cv2.putText(img,str(goptag+gopcont),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        if img is None:
            break
        if gopicont >=1:
            goptag = not(goptag)
            gopicont =0
        if gopcont >=maxgop:
            goptag = not(goptag)
            gopcont = 0
        #tic = time.time()
        img0 = img.copy()
        if ((not(goptag))or(maxgop==0)):
        #if True:    
            nboxes = []
            nconfs = []
            nclss = []
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            boxes[:,2:4] -= boxes[:,0:2]
            for a in range(len(clss)):
                if clss[a] == 0:
                    nclss.append(clss[a])
                    nboxes.append(np.array(boxes[a]))
                    nconfs.append(confs[a])
            #nclss = np.array(nclss)
            #nboxes = np.array(nboxes)
            #nconfs = np.array(nconfs)
            dcont += 1     
            gopicont +=1
            em0 = time.time()
            encodresult = encode(img,nboxes)
            em1 = time.time()           
        if goptag:
            gopcont +=1
        pkg = []
        tdc = time.time()
        if len(nboxes)>=1:     
        #if False:
            subcont += 1

            #print(encodresult)
            #multitracker,cnnt,fmt,pdt,upt,mht = trackers.startrack(nboxes,img,nconfs,not(goptag))
            #multitracker,cnnt,fmt,pdt,upt,mht,mtx = trackers.startrack1(nboxes,img,nconfs,not(goptag),encodresult)
            multitracker,PDT = trackers.startrack1(nboxes,img,nconfs,(not(goptag)or(maxgop==0)),encodresult)
            #multitracker = trackers.startrack(nboxes,img,nconfs,True)
            for res in multitracker:
                if not res.is_confirmed() or res.time_since_update > 0:
                    continue
                subbox0 = list(res.to_tlbr())
                #print(subbox0)
                subpkg= subbox0
                subpkg.append(res.conf)
                subpkg.append(res.track_id)
                subpkg.append(0)
                pkg.append((subpkg))
            #toc = time.time()
            pkg = np.array(pkg)
            iboxes = pkg[:,0:4]
            iconfs = pkg[:,4]
            iclss = pkg[:,5]
            iclss2 = pkg[:,6]
            img0 = vis.draw_bboxes(img0, iboxes, iclss, iclss2)
            toc = time.time()
            #if True:
            if cont >(maxgop+1):
                tim +=(toc-tic)
                fps0 = 1/(tim/(totalcont-(maxgop+1)))
                #fps = 1/(tim/(cont-(maxgop+1)))
                #fps = 1/(tim/(cont))
                img0 = show_fps(img0, fps)

            #print("iou",mtx)
                #cv2.putText(img0,fps, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                #img0 = show_fps(img0, fps)
        img0 = cv2.resize(img0,(640,480))
        img0 = np.copy(img0)
        #cv2.imshow(WINDOW_NAME, img0)
        #out.write(img0)
        #print(mtx)
        #curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        #fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        #tic = toc
        #if cont != 1:
        #with open("./output/"+str(totalcont)+ ".txt","w+") as f:
        #    if len(nboxes)>=1:
        #        dec = [str(totalcont)+" "+str(tim/subcont)+" "+str(det/subcont)+" "+str(ct/subcont)+" "+str(ft/subcont)+" "+str(pt/subcont)+" "+str(ut/subcont)+" "+str(mt/subcont)]
        #        f.write(str(dec[0]))
        
        #if cont ==(maxgop+1)*10:
        #    dcont,subcont = 0,0
            #cont,dcont,subcont = 0,0,0
        #    tim ,ct,ft,pt,ut,dt,det,mt= 0,0,0,0,0,0,0,0
        #GETpath = "./testmapagilev3l416bbb2turnnewmotion0101/"
       
       
#write turn and stright
        if pframe ==0:
            with open(GETpath+"p"+str(pframe)+"/"+outname,"w+") as f:
                for ms in pkg:
                    dec = ["0"+" "+str(ms[0])+" "+str(ms[1])+" "+str(ms[2])+" "+str(ms[3])+" "+"\n"]
                    f.write(dec[0])
            with open(GETpath+"p"+str(pframe+1)+"00/"+outname,"w+") as f:
                for ms in pkg:
                    dec = ["0"+" "+str(ms[4])+" "+str(ms[0])+" "+str(ms[1])+" "+str(ms[2])+" "+str(ms[3])+" "+"\n"]
                    f.write(dec[0])
        else:
            with open(GETpath+"p"+str(pframe)+"/"+outname,"w+") as f:
                for ms in pkg:
                    dec = ["0"+" "+str(ms[4])+" "+str(ms[0])+" "+str(ms[1])+" "+str(ms[2])+" "+str(ms[3])+" "+"\n"]
                    f.write(dec[0])
            with open(GETpath+"n"+str(pframe)+"/"+outname,"w+") as f:
                for a in range(len(nboxes)):
                    dec = ["0"+" "+str(nconfs[a])+" "+str(nboxes[a][0])+" "+str(nboxes[a][1])+" "+str(nboxes[a][2]+nboxes[a][0])+" "+str(nboxes[a][3]+nboxes[a][1])+" "+"\n"]
                    f.write(dec[0])
        key = cv2.waitKey(1)
        if (key == 27) :  # ESC key: quit program
            print(fps)
            break
        elif key == ord('c'):
            tim =0
            cont = 0
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
    print(fps0)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    
    trackers = tracker(args.tracker_model,max_cosine_distance=0.2,nn_budget = 3)
    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(416)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
    
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    #encode = gdet.create_box_encoder("deepsort-fp16-bach1.trt",batch_size=1)
    encode = gdet.create_box_encoder("/home/soc507/tftotrt/zawamodeltrain",batch_size=1)
    #open_window(
    #    WINDOW_NAME, 'Camera TensorRT YOLO Demo',
    #    cam.img_width, cam.img_height)
    for p in range(11):
        trackers.reset()
        cam = Camera(args)
        if not cam.isOpened():
            raise SystemExit('ERROR: failed to open camera!')
        vis = BBoxVisualization(cls_dict)
        loop_and_detect(cam, trt_yolo, conf_th=0.40, vis=vis,trackers=trackers,pframe=p,encode=encode)
        
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
