# collect the tbbrrayracker
import numpy as np 


def takespeedV2mean(track,fps):
    speed = []
    if len(track)>2:
        for i in range(len(track)-2):
            # subspeed =[]
            dx0 = track[i][0]
            dx2 = track[i+2][0]
            # print("dx",dx0,dx2)
            # print("fps",fps)
            speed.append((((dx2-dx0))*fps)/2)
    elif len(track)>0:
        dx0 = track[0][0]
        dx2 = track[1][0]
        # print("dx",dx0,dx2)
        # print("fps",fps)
        speed.append((((dx2-dx0))*fps)/2)
    else:
        speed.append(0)
    return speed 
class contstar():
    def __init__(self,time,h,w):
        self.conter = []
        self.timesave = time
        self.w = w
        self.h = h
    def saver(self,bboxs):
        self.conter.append(bboxs)
        if len(self.conter) > self.timesave:
            self.conter.pop(0)
    def tracklines(self,tracknum):
        trline = []
        dxline = []
        area = []
        final = self.conter[-1][(self.conter[-1][:,5])==int(tracknum)]
        if len(final)>0:
            final = final[0]
            dfx = final[2]-final[0]
            dfy = final[3]-final[1]
        for n in range(len(self.conter)):
            if len(self.conter[n]) == 0:
                continue
            subtrack = self.conter[n][(self.conter[n][:,5])==int(tracknum)]
            if len(subtrack)>0:
                subtrack = subtrack[0]
                x = subtrack[2]+subtrack[0]
                dx = subtrack[2]-subtrack[0]
                y = subtrack[3]+subtrack[1]
                dy = subtrack[3]-subtrack[1]
                subtr = []  
                dxline.append(dx)              
                subtr.append(int(x/2))
                subtr.append(int(y/2))
                trline.append(subtr)
                subarea = dx*dy
                area.append(subarea)
        return trline,area,dxline
    def therule(self,line,time,weigh,area,dx):
        nx1 = line[-1][0]-(self.w/2)
        dx1 =[nx1-dx[-1]*0.5,nx1+dx[-1]*0.5]
        nx0 = line[(int(len(line)/2)-1)][0]-(self.w/2)
        # dx0 =[line[-2][0]-(self.w/2)-dx[-2]*0.5,line[-2][0]-(self.w/2)+dx[-2]*0.5]
        dx0 =[nx0-dx[(int(len(line)/2)-1)]*0.5,nx0+dx[(int(len(line)/2)-1)]*0.5]
        movedx = [dx1[0]-dx0[0],dx1[1]-dx0[1]]
        # movearea = abs((area[(len(line)/2-1)])-area[-1])/(area[(len(line)/2-1)])
        movearea = (max(area[int(len(line)*2/3-1):])-area[-1])/max(area[int(len(line)*2/3-1):])
        clo = 0
        weigh0 = dx[-1]
        th = 0.06
        bm = 1
        if (area[-1] >= 0000)and(movearea<0.4):
            if abs(nx1-nx0)>(0.06*weigh0):  
                # if ((nx1-nx0)<0) and ((movedx[0])<(bm*th*weigh0)):
                if ((nx1-nx0)<0) and ((movedx[0])<(bm*th*weigh0)):
                    if dx0[1]*((dx1[1]+nx1)/2) >= 0:
                        if abs(nx1) < abs(nx0):                       
                            speed = takespeedV2mean(line,time)

                            if  speed[-1]>= 400:
                                clo = 1
                            else:
                                clo = 1
                elif ((nx1-nx0)>0)and ((movedx[1])>(bm*th*weigh0)):
                # elif ((nx1-nx0)>0) and ((movedx[1])>(bm*th*weigh0)):
                    if dx0[0]*((dx1[0]+nx1)/2) >= 0:
                        if abs(nx1) < abs(nx0):                       
                            speed = takespeedV2mean(line,time)
                            if  speed[-1]>= 400:
                                clo = 1
                            else:
                                clo = 1
        return clo
