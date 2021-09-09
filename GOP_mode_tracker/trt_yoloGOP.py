"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO
from testdpio import tracker

from deep_sort import generate_detectionsTRT as gdet
WINDOW_NAME = 'TrtYOLODemo'
from rememberV2 import contstar 
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
        '-t', '--tracker_model', type=str,default ='./deepsort-fp16-bach1.trt',
        help='loading reid model')
    
    parser.add_argument(
        '-p', '--pframeset', type = int,default = 6,
        help='set the pframe')
    '''
    parser.add_argument(
        '-p', '--p_frame_num', type=int, default=0,
        help='GOP mode : input number of p-frame, default=0.')
    '''
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis,trackers):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.

    p-frame : p_frame_num, now_times, input_trt_yolo, input_vis .

    """
    lines = contstar(20,480,640)
    full_scrn=False
    cont = 0
    tim = 0
    fps = 0.0
    goptag = 0
    gopcont = 0
    gopicont = 0
    totalcont = 0
    det = 0
    dcont = 0
    subcont = 0
    maxgop = 3 #MAX GOP frame
    stop = 1
    catch = False
    out = cv2.VideoWriter("./papertestmoontest.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
    encode = gdet.create_box_encoder("/home/soc507/tftotrt/zawamodeltrain", batch_size=1)
    while True:
        cont +=1
        totalcont += 1
        outname = str(totalcont).rjust(5,'0')+".txt"
        img = cam.read() 
        img = cv2.resize(img,(640,480))  
  
        tic = time.time()
        if img is None:
            break
        if gopicont >=1:
            goptag = not(goptag)
            gopicont =0
        if gopcont >=maxgop:
            goptag = not(goptag)
            gopcont = 0
        tic = time.time()
        img0 = img.copy()
        if (not(goptag)or(maxgop==0)):
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
            dcont += 1     
            gopicont +=1
            em0 = time.time()
            encodresult = encode(img,nboxes)
            print(encodresult.shape)
            em1 = time.time()           
        if goptag:
            gopcont +=1
        
        tdc = time.time()
        pkg = []
        if len(nboxes)>=1:     
            subcont += 1
            multitracker,pdt = trackers.startrack1(nboxes,img,nconfs,(not(goptag)or(maxgop==0)),encodresult)
            for res in multitracker:
                print(len(pkg))
                if not res.is_confirmed() or res.time_since_update > 0:
                    continue
                subbox0 = list(res.to_tlbr())
                boxweight = subbox0[2]-subbox0[0]
                subpkg= subbox0
                subpkg.append(res.conf)
                subpkg.append(res.track_id)
                subpkg.append(0)
                pkg.append((subpkg))
            #toc = time.time()
            pkg = np.array(pkg)
            #lines.saver(pkg)
            iboxes,iconfs,iclss,iclss2  = [],[],[],[]
            if len(pkg) >0: 
                print(len(pkg))
                iboxes = pkg[:,0:4]
                iconfs = pkg[:,4]

                iclss = pkg[:,5]
                iclss2 = pkg[:,6]
                 
                appcol = []
            #for ids in iclss:
            #    line,area,dx = lines.tracklines(int(ids))
            #    #print(lines.conter)
            #    #print("line",line)
            #    if len(dx)>=1:
            #        apclass = lines.therule(line,30,boxweight,area,dx)
            #    #    print(apclass)
            #    else:
            #        apclass = 0
            #    appcol.append(apclass)
                img0 = vis.draw_bboxes(img0, iboxes, iclss, iclss2)
            #img0 = vis.draw_bboxes(img0, iboxes, appcol, iclss2)
        toc = time.time()
      
        if totalcont > (maxgop+1):
            tim +=(toc-tic)

            fps0 = 1/(tim/(cont))

            img0 = show_fps(img0, fps)

        img0 = cv2.resize(img0,(640,480))
        img0 = np.copy(img0)
        cv2.imshow(WINDOW_NAME, img0)
        
        out.write(img0)

        if cont == 10 *(maxgop+1):
            fps = fps0
            cont,dcont,subcont = 0,0,0
            tim,ct,ft,pt,ut,dt,det= 0,0,0,0,0,0,0
        key = cv2.waitKey(int(stop))
        if (key == 27):  # ESC key: quit program
            print(fps)
            break
        elif key == ord('c'):
            #stop = not(stop)
            catch = True
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
    trackers = tracker(args.tracker_model,max_cosine_distance=0.5,nn_budget = 1)
    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]
    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
    
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)

    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis,trackers=trackers)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
