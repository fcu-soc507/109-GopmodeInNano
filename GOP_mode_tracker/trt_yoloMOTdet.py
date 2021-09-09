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

from deep_sort import generate_detectionsTRT1 as gdet
WINDOW_NAME = 'TrtYOLODemo'
from rememberV2 import contstar
#import readmot as MT
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
        '-p', '--pframeset', type = int,default = 6,
        help='set the pframe')
    '''
    parser.add_argument(
        '-p', '--p_frame_num', type=int, default=0,
        help='GOP mode : input number of p-frame, default=0.')
    '''
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis,trackers,ipath,tt,maxgop):
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
    tim0 = 0
    trtim = 0
    fps = 0.0
    goptag = 0
    gopcont = 0
    gopicont = 0
    totalcont = 0
    idt = 0
    iet = 0
    det = 0
    dcont = 0
    subcont = 0
    resize = 0
    TCTG,TINF,TGTC,EBT,INSET,OUTSET = 0,0,0,0,0,0
    #maxgop = 3 #MAX GOP frame
    stop = 1
    mode = "w+"
    #out = cv2.VideoWriter("./testout.mp4",cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
    path = os.listdir(ipath)
    path.sort(key=lambda x:int(x[2:-4]))
    #encode = gdet.create_box_encoder("./deepsort-fp16-bach1.trt", batch_size=1)
    encode = gdet.create_box_encoder("/home/soc507/tftotrt/zawamodeltrain", batch_size=1)
    #encode = gdet.create_box_encoder("/home/soc507/tftotrt/smallmodeltest", batch_size=1)
    #encode = gdet.create_box_encoder("./mars-small128.pb", batch_size=1)
    #encode = gdet.create_box_encoder("./marsV4-IFs.pb", batch_size=1)
    #encode = gdet.create_box_encoder("./marsV1-1.pb", batch_size=1)
    #for imgname in path:
    #    cam = cv2.VideoCapture("./input/"+imgname)
    
    for fnum in path:
        cam = cv2.VideoCapture(ipath+fnum)
        #print(ipath+fnum)
        cont +=1
        totalcont += 1
        outname = str(totalcont).rjust(5,'0')+".txt"
        #if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #    break
        ret,img = cam.read() 
        imgsh=np.shape(img)
        hr = imgsh[0]/480
        wr = imgsh[1]/640
        img = cv2.resize(img,(640,480))  
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
        tic = time.time()
        img0 = img.copy()
        if (not(goptag)or(maxgop==0)):
        #if True:    
            itic = time.time()
            nboxes = []
            nconfs = []
            nclss = []
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            boxes[:,2:4] -= boxes[:,0:2]
            itic0 = time.time()
            for a in range(len(clss)):
                if clss[a] == 0:
                    nclss.append(clss[a])
                    nboxes.append(np.array(boxes[a]))
                    nconfs.append(confs[a])
            #nclss = np.array(nclss)
            #nboxes = np.array(nboxes)
            #nconfs = np.array(nconfs)
            dcont += 1     
            #print(boxes) 
            gopicont +=1
            em0 = time.time()
            #print("-------------")
            #print(itic0-itic)
            #print(em0-itic)
            encodresult,times001,CTG,INF,GTC,a,b,c = encode(img,nboxes)
            em1 = time.time()           
        if goptag:
            gopcont +=1
        
        #print(boxes)
        tdc = time.time()
        pkg = []
        if len(nboxes)>=1:     
        #if False:
            subcont += 1
            #print(encodresult)
            #multitracker,cnnt,fmt,pdt,upt,mht = trackers.startrack(nboxes,img,nconfs,not(goptag))
            #multitracker,cnnt,fmt,pdt,upt,mht,mtx = trackers.startrack1(nboxes,img,nconfs,(not(goptag)or(maxgop==0)),encodresult)
            multitracker,pdt = trackers.startrack1(nboxes,img,nconfs,(not(goptag)or(maxgop==0)),encodresult)
            #multitracker,cnnt,fmt,pdt,upt,mht,mtx = trackers.startrack1(nboxes,img,nconfs,True,encodresult)
            #multitracker = trackers.startrack(nboxes,img,nconfs,True)
            outtime = time.time()
            for res in multitracker:
                if not res.is_confirmed() or res.time_since_update > 0:
                    continue
                subbox0 = list(res.to_tlbr())
                boxweight = subbox0[2]-subbox0[0]
                #print(subbox0)
                subpkg= subbox0
                subpkg.append(res.conf)
                subpkg.append(res.track_id)
                subpkg.append(0)
                pkg.append((subpkg))
            #toc = time.time()
            pkg = np.array(pkg)
            #lines.saver(pkg)
            if len(pkg) >0 :
                iboxes = pkg[:,0:4]
                iconfs = pkg[:,4]

                iclss = pkg[:,5]
                iclss2 = pkg[:,6]
            else:
                iboxes = []
                iconfs = []
                iclss = []
                iclss2 = []
            appcol = []
            #for ids in iclss:
            #    line,area,dx = lines.tracklines(int(ids))
            #    #print(lines.conter)
                #print("line",line)
            #    if len(dx)>=1:
            #        apclass = lines.therule(line,30,boxweight,area,dx)
                #    print(apclass)
            #    else:
            #        apclass = 0
            #    appcol.append(apclass)
            toc = time.time()
            #img0 = vis.draw_bboxes(img0, iboxes, iclss, iclss2)

            img0 = vis.draw_bboxes(img0, iboxes, iclss, iconfs)
            #img0 = vis.draw_bboxes(img0, iboxes, appcol, iclss2)
            #if True:
            if totalcont > (maxgop+1):
                tim0 = (toc-em0)
                tim +=(toc-tic)
                idt +=(em0-itic)
                iet +=(em1-em0)
                trtim +=(toc-tdc)
                resize += times001
                TCTG +=CTG
                TINF +=INF
                TGTC +=GTC
                EBT += a
                INSET +=b
                OUTSET +=c
                fps0 = 1/(tim/(totalcont-(maxgop+1)))
                #print("fps",fps0)
        img0 = cv2.resize(img0,(640,480))
        img0 = np.copy(img0)
        #cv2.imshow(WINDOW_NAME, img0)
        with open("./MPyolo3p"+str(maxgop)+"/"+tt+".txt",mode)as f:
            for ms in pkg:
                dec = ["%d,-1,%.2f,%.2f,%.2f,%.2f,%.2f\n"%(cont,(ms[0])*wr,(ms[1])*hr,(ms[2]-ms[0])*wr,(ms[3]-ms[1])*hr,ms[4])]
                f.write(str(dec[0]))
        mode = "a+"
        #out.write(img0)
        #if cont ==70:
        #    dcont,subcont = 0,0
        #    ct,ft,pt,ut,dt,det,mt= 0,0,0,0,0,0,0
        key = cv2.waitKey(int(stop))
        if (key == 27) :  # ESC key: quit program
            print(fps)
            break
        elif key == ord('c'):
            stop = not(stop)
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
    #cv2.destroyWindow('')
    print("-------------------")
    print('the time',tim)  
    print("detime",idt)
    print("emtime",iet)
    print("resize",resize)
    print("prd&updata",trtim)
    print("CTG",TCTG)
    print("INF",TINF)
    print("GTC",TGTC)
    print("EBT",EBT)
    print("INSET",INSET)
    print("outset",OUTSET)
    return tim ,idt,iet,trtim,tim0,subcont,resize,TCTG,TINF,TGTC
def main():
    path = "/home/soc507/MOT17Det/"
    #mode = ["train/"]
    #mode = ["train/","test/"]
    mode = ["test/"]
    #pset = [0,3,6,9]
    pset = [0]
    totaltims = []
    totalidt = []
    totaliet = []
    totaltrtim = []
    imgpath = "/img1/"
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    
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
    
    cam = Camera(args)
#   open_window(WINDOW_NAME, 'Camera TensorRT YOLO Demo',
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
        #cam.img_width, cam.img_height)
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num, args.letter_box)
    for ps in pset:
        tims,idts,iets,trtims,alldcont,resis,CTG,INF,GTC =0,0,0,0,0,0,0,0,0

        for mo in mode:
            
            totalpath = os.listdir(path+mo)
            totalpath.sort()
            for a in totalpath:
                trackers.reset()
                suba = path+mo+a+imgpath
                vis = BBoxVisualization(cls_dict)
                #loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis,trackers=trackers,ipath = suba,tt = a ,maxgop = ps)
                tim ,idt,iet,trtim,tim0,dcont,sis,a,b,c= loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis,trackers=trackers,ipath = suba,tt = a ,maxgop = ps)
                tims+=tim
                idts+=idt
                iets+=iet
                trtims += trtim
                resis += sis
                alldcont += dcont
                CTG+=a
                INF+=b
                GTC+=c
        totaltims.append(tims)
        totalidt.append(idts)
        totaliet.append(iets)
        totaltrtim.append(trtim)
    cam.release()
    cv2.destroyAllWindows()
    
    print("total",totaltims)
    print("det",totalidt)
    print("emb",totaliet)
    print("U&P",totaltrtim)
    print("dcont",alldcont)
    print("resize",resis)
    print("CTG",CTG)
    print("INF",INF)
    print("GTC",GTC)
    with open("motnewourtrt04.txt","w+")as mmf:
        
        for a in totaltims:
            mmf.write(str(a)+"\n")


if __name__ == '__main__':
    main()
