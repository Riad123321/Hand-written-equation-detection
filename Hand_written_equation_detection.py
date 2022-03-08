!add-apt-repository -y ppa:alex-p/tesseract-ocr-devel
!apt-get update
!apt-get install tesseract-ocr
!pip install pytesseract

import pytesseract 
import cv2
import numpy as np
from imutils.contours import sort_contours
from google.colab.patches import cv2_imshow
cap = cv2.VideoCapture(Video Path) #path to your video
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,10))
import imutils

if (cap.isOpened() == False):
  print("Unable to read camera feed")
  
options = "--psm 8"
out = cv2.VideoWriter('equation_detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (540,380))   

while (True):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                            interpolation = cv2.INTER_CUBIC)

        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        gray = np.array(gray,dtype='uint8')

        thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        dist = cv2.distanceTransform(thresh,cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist.copy(),dist.copy(),0,1.0,cv2.NORM_MINMAX)
        dist_norm = (dist_norm * 255).astype("uint8")
        dist_norm = cv2.threshold(dist_norm,0,255,
                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            

        dist_closing = cv2.morphologyEx(dist_norm.copy(),cv2.MORPH_CLOSE, rectKernel)
        dist_closing_cnts = cv2.findContours(dist_closing.copy(),cv2.RETR_EXTERNAL
                                      ,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(dist_closing_cnts)
        Hight,Width = frame.shape[:2]
        texts = []
        for cnt in cnts:
          (x,y,w,h) = cv2.boundingRect(cnt)
          if w >= 13 and h >= 13:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            detected_part = dist_norm[max(y-13,0):min(y+h+13,Hight-1),max(x-13,0):min(x+w+13,Width-13)]
            part_pred = pytesseract.image_to_string(detected_part,config=options)
            if part_pred != '':
                texts.append(str(part_pred.strip()))
        OK = [True for x in texts if x.lower() == 'ok']
        
        yes_OK = False
        for extracted_OK in OK:
          if extracted_OK == True:
            yes_OK = True
            break
        cv2.rectangle(frame,(0,0),(136,136),(255,255,255),-1)
        cv2.rectangle(frame,(0,0),(135,135),(0,0,0),2)

        if yes_OK==False:
          frame = cv2.putText(frame, 'No Equation', (13,65), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0,0,255), 2)
        else:
          all_parts = []
          equation = []
          cnts_y_axis = []
          for c in cnts:
                (x,y,w,h) = cv2.boundingRect(c)
                if w >= 15 and h >= 15:#you can change it as you want
                  all_parts.append(c)
                  cnts_y_axis.append(cv2.boundingRect(c)[1])
          idx = np.argpartition(cnts_y_axis,2)
          equation = [cnt for n,cnt in enumerate(all_parts) if n not in idx[:2]]
          equation = np.vstack([equation[i] for i in range(0, len(equation))])
          hull = cv2.convexHull(equation)    
          (x,y,w,h) = cv2.boundingRect(hull)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1) 
          detected_part = dist_norm[max(y-10,0):min(y+h+10,Hight-1),max(x-10,0):min(x+w+10,Width-1)] 
          part_pred = pytesseract.image_to_string(detected_part,config=options)
          x = np.arange(-68,68) 
          try:
              cv2.rectangle(frame,(0,0),(68,68),(0,0,0),1)
              cv2.rectangle(frame,(68,68),(135,135),(0,0,0),1)
              y = eval(part_pred.lower())
              x_new =  x + 68
              y_new =  np.array((-1*y/3)+ 68,dtype='int32')
              xy_array = np.concatenate([x_new.reshape(-1,1),y_new.reshape(-1,1)],axis=1)
              xy_array = xy_array[np.array(y_new) < 135]
              xy_array = xy_array.reshape((-1,1,2))
              cv2.polylines(frame,[xy_array],True,(0,0,255),thickness=2)
          except:
              frame = cv2.putText(frame, 'Error', (13,65), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, (0,0,255), 2)       
          #cv2.rectangle(frame,(max(np.min(x),0),max(np.min(y),0)),(min(np.max(x),Width-1),min(np.max(y),Hight-1)),(0,0,255),1)
          for part in all_parts:
              (x,y,w,h) = cv2.boundingRect(part)
              cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        out.write(frame)
    else:
          break    
cap.release()
out.release()
