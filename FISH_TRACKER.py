#pip install opencv-python==4.4.0.46
#pip install opencv-contrib-python==4.4.0.46

import numpy as np
import pandas as pd
import sys
import cv2
from random import randint
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import time

#excel saver
def excelUpdater(cPosition_1,cPosition_2,cPosition_3,tPosition_1,tPosition_2,tPosition_3):
    #length to make all equal
    max_length = max(len(cPosition_1), len(cPosition_2), len(cPosition_3), len(tPosition_1), len(tPosition_2), len(tPosition_3))

    # Create NumPy arrays with missing values
    cPosition_1= np.array(cPosition_1 + [None] * (max_length - len(cPosition_1)))
    cPosition_2= np.array(cPosition_2 + [None] * (max_length - len(cPosition_2)))
    cPosition_3= np.array(cPosition_3 + [None] * (max_length - len(cPosition_3)))
    tPosition_1= np.array(tPosition_1 + [None] * (max_length - len(tPosition_1)))
    tPosition_2= np.array(tPosition_2 + [None] * (max_length - len(tPosition_2)))
    tPosition_3= np.array(tPosition_3 + [None] * (max_length - len(tPosition_3)))
    
    # Create a structured NumPy array
    data = np.column_stack((cPosition_1, cPosition_2, cPosition_3, tPosition_1, tPosition_2, tPosition_3))

    # Create a DataFrame from the structured NumPy array
    df = pd.DataFrame(data, columns=['cPosition_1', 'cPosition_2','cPosition_3', 'tPosition_1','tPosition_2','tPosition_3'])

    # Define the Excel filename
    excel_filename = "data_columnwise.xlsx"

    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)


#tracker caller
def trackerCreator(trackerType):
    if(trackerType==trackerTypes[0]):
        tracker=cv2.TrackerCSRT_create()
    elif(trackerType==trackerTypes[1]):
        tracker=cv2.TrackerMIL_create()
    else:
        tracker=None
        print("Incorrect tracker selected!")
        print("Available trackers are : \n 1) CSRT \n 2) MIL")
    return tracker

def fish(cap):
    cap=cap
    width  = cap.get(3)  # width of video
    height = cap.get(4)  # height of video
    half_height=height/1.7 #to draw center of water tank
    vertical_point=height*0.09 #to get glass
    vertical_mirror_line_left=width/6 #left glass
    vertical_mirror_line_right=width/1.06 #right glass
    interval=int(input("Interveral to capture the XY positions in seconds")) #interval in seconds

    #boxes and colors
    bboxes = []
    colors = []

    #to draw 3d graph where z is always 0
    X1=[]
    Y1=[]
    X2=[]
    Y2=[]
    X3=[]
    Y3=[]

    img = np.zeros((int(height),int(width),3), np.uint8)

    #start
    ret, frame=cap.read() #reading frame
    if not ret:
        print("Failed to read the video!")
        sys.exit(1)

    fish=int(input("How many fish to be tracked? Min 1 - Max 3"))
    if(fish<1):
        fish=1
    if(fish>3):
        fish=3
    print(fish)
    
    selector=0
    #ROI(region of interest) selector
    while True:
        bbox=cv2.selectROI('Multi Fish Tracker',frame)
        bboxes.append(bbox)
        colors.append((randint(0,255),randint(0,255),randint(0,255)))
        selector+=1
        print("press enter")
        if(selector==fish):
            break
    cv2.destroyAllWindows()
        
    print("Selected bounding boxes are {}".format(bboxes))

    #Tracker type
    trackerType=trackerTypes[0]
    print("You've selected {} tracker".format(trackerType))

    #Multitracker
    multiTracker=cv2.MultiTracker_create()

    #Initialize multiTracker
    for bbox in bboxes:
        multiTracker.add(trackerCreator(trackerType),frame,bbox)

    counter_up_1=0 #fish above tank
    counter_down_1=0 #fish below tank
    counter_center_1=0 #fish center tank
    
    counter_up_2=0
    counter_down_2=0
    counter_center_2=0

    counter_up_3=0
    counter_down_3=0
    counter_center_3=0

    counter_left_1=0 #fish left tank
    counter_right_1=0 #fish right tank
    counter_mid_1=0 #fish mid tank

    counter_left_2=0
    counter_right_2=0
    counter_mid_2=0

    counter_left_3=0
    counter_right_3=0
    counter_mid_3=0

    
    count_up_1=True #is fish above tank?
    count_down_1=True #is fish below tank?
    count_center_1=True #is fish center tank?

    count_up_2=True
    count_down_2=True
    count_center_2=True

    count_up_3=True
    count_down_3=True
    count_center_3=True

    count_left_1=True #is fish left tank?
    count_right_1=True #is fish right tank?
    count_mid_1=True #is fish mid tank?

    count_left_2=True
    count_right_2=True
    count_mid_2=True

    count_left_3=True
    count_right_3=True
    count_mid_3=True

    up_counter_1=0 #string format up
    down_counter_1=0 #string format down
    center_counter_1=0 #string format center

    up_counter_2=0
    down_counter_2=0
    center_counter_2=0

    up_counter_3=0
    down_counter_3=0
    center_counter_3=0

    left_counter_1=0 #string format left
    right_counter_1=0 #string format right
    mid_counter_1=0 #string format mid

    left_counter_2=0
    right_counter_2=0
    mid_counter_2=0

    left_counter_3=0
    right_counter_3=0
    mid_counter_3=0
    

    up_timer_1=0 #fish above the line time
    down_timer_1=0 #fish below the line time
    center_timer_1=0 #fish center the line time

    up_timer_2=0
    down_timer_2=0
    center_timer_2=0

    up_timer_3=0
    down_timer_3=0
    center_timer_3=0
    
    left_timer_1=0 #fish left the line time
    right_timer_1=0 #fish right the line time
    mid_timer_1=0 #fish mid the line time

    left_timer_2=0
    right_timer_2=0
    mid_timer_2=0

    left_timer_3=0
    right_timer_3=0
    mid_timer_3=0

    
    position_1=[] #mean position

    position_2=[]
    
    position_3=[]
    
    starting_time=0 #world starting time
    global_time=0 #video playback time
    starting_time = time.time() #world tarting time
    start_time=time.time() #n seconds counter
    frame_count = 0

    #Process video and track objects
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break

        #drawing horizontal center line  #blue #red
        cv2.line(frame, pt1=(0,int(half_height)-int(vertical_point)), pt2=(int(width*0.05),int(half_height)-int(vertical_point)), color=(0,0,255), thickness=1)
        cv2.line(img, pt1=(0,int(half_height)-int(vertical_point)), pt2=(int(width*0.05),int(half_height)-int(vertical_point)), color=(0,0,255), thickness=1)
        cv2.line(frame, pt1=(0,int(half_height)+int(vertical_point)), pt2=(int(width*0.05),int(half_height)+int(vertical_point)), color=(255,0,0), thickness=1)
        cv2.line(img, pt1=(0,int(half_height)+int(vertical_point)), pt2=(int(width*0.05),int(half_height)+int(vertical_point)), color=(255,0,0), thickness=1)        
        #drawing vertical mirror line #yellow #cyan
        cv2.line(frame, pt1=(int(vertical_mirror_line_left),0), pt2=(int(vertical_mirror_line_left),int(height*0.05)), color=(255,255,0), thickness=1)
        cv2.line(img, pt1=(int(vertical_mirror_line_left),0), pt2=(int(vertical_mirror_line_left),int(height*0.05)), color=(255,255,0), thickness=1)
        #drawing vertical mirror line
        cv2.line(frame, pt1=(int(vertical_mirror_line_right),0), pt2=(int(vertical_mirror_line_right),int(height*0.05)), color=(0,255,255), thickness=1)        
        cv2.line(img, pt1=(int(vertical_mirror_line_right),0), pt2=(int(vertical_mirror_line_right),int(height*0.05)), color=(0,255,255), thickness=1)        

        #timer text
        cv2.putText(frame, "Timer = {}".format(global_time), (int(width*0.45), int(height-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)
        
        #get updated location of objects in subsequent frames
        ret,boxes=multiTracker.update(frame)

        # Increment the frame count
        frame_count += 1
        current_time = time.time()

        try:
            #draw tracked objects
            for i, newbox in enumerate(boxes):



                #IF FISH 1 EXISTS
                if(i==0):
                    cv2.putText(frame, "F1C (TOP/CENTER/BOTTOM - LEFT/MID/RIGHT) = {}/{}/{} - {}/{}/{}".format(up_counter_1,center_counter_1,down_counter_1,left_counter_1,mid_counter_1,right_counter_1), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)
                    cv2.putText(frame, "F1T (TOP/CENTER/BOTTOM - LEFT/MID/RIGHT) = {}/{}/{} - {}/{}/{}".format(up_timer_1,center_timer_1,down_timer_1,left_timer_1,mid_timer_1,right_timer_1), (int(width*0.50), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)

                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    c1=int((p1[0]+p2[0])/2)
                    c2=int((p1[1]+p2[1])/2)
                    
                    X1.append(c1)
                    Y1.append(c2)
                    
                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                    #track on black image
                    cv2.circle(img,(c1,c2),2,colors[i],-1)


                    #mid timer
                    if(c2>=int(vertical_mirror_line_left) and c1<=int(vertical_mirror_line_right)):
                       mid_timer_1=int(global_time-(left_timer_1+right_timer_1))
                    #left timer
                    elif(c2<int(vertical_mirror_line_left)):
                        left_timer_1=int(global_time-(right_timer_1+mid_timer_1))
                    #right timer
                    elif(c2>int(vertical_mirror_line_right)):
                        right_timer_1=int(global_time-(left_timer_1+mid_timer_1))
                        

                    #is fish mid the tank? how many times?
                    if(c2>=int(vertical_mirror_line_left) and c1<=int(vertical_mirror_line_right) and count_mid_1==True):
                        counter_mid_1+=1
                        count_mid_1=False
                        count_left_1=True
                        count_right_1=True
                        mid_counter_1=int(counter_mid_1)
                    #is fish left the tank? how many times?
                    elif(c2<int(vertical_mirror_line_left) and count_left_1==True):
                        counter_left_1+=1
                        count_left_1=False
                        count_mid_1=True
                        count_right_1=True
                        left_counter_1=int(counter_left_1)
                    #is fish right the tank? how many times?
                    elif(c2>int(vertical_mirror_line_right) and count_right_1==True):
                        counter_right_1+=1
                        count_right_1=False
                        count_left_1=True
                        count_mid_1=True
                        right_counter_1=int(counter_right_1)
                        

                    #center timer
                    if(c2>=(int(half_height)-int(vertical_point)) and c2<=(int(half_height)+int(vertical_point))):
                        center_timer_1=int(global_time-(up_timer_1+down_timer_1))
                    #above timer
                    elif(c2<(int(half_height)-int(vertical_point))):
                        up_timer_1=int(global_time-(down_timer_1+center_timer_1))
                    #below timer
                    elif (c2>(int(half_height)+int(vertical_point))):
                        down_timer_1=int(global_time-(up_timer_1+center_timer_1))

                        
                    #is fish center the tank? how many times?
                    if(c2>=(int(half_height)-int(vertical_point)) and c2<=(int(half_height)+int(vertical_point)) and count_center_1==True):
                        counter_center_1+=1
                        count_center_1=False
                        count_up_1=True
                        count_down_1=True
                        center_counter_1=int(counter_center_1)
                    #is fish above the tank? how many times?
                    elif(c2<(int(half_height)-int(vertical_point)) and count_up_1==True):
                        counter_up_1+=1
                        count_up_1=False
                        count_center_1=True
                        count_down_1=True
                        up_counter_1=int(counter_up_1)
                    #is fish below the tank? how many times?
                    elif(c2>(int(half_height)+int(vertical_point)) and count_down_1==True):
                        counter_down_1+=1
                        count_down_1=False
                        count_up_1=True
                        count_center_1=True
                        down_counter_1=int(counter_down_1)


                    #append position
                    if current_time - start_time >= interval:
                        position_1.append(int(c1))
                        print(position_1)
                        if(selector==1):
                            start_time = current_time



                #IF FISH 2 EXISTS 
                if(i==1):
                    cv2.putText(frame, "F2C (TOP/CENTER/BOTTOM - LEFT/MID/RIGHT) = {}/{}/{} - {}/{}/{}".format(up_counter_2,center_counter_2,down_counter_2,left_counter_2,mid_counter_2,right_counter_2), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)
                    cv2.putText(frame, "F2T (TOP/CENTER/BOTTOM - LEFT/MID/RIGHT) = {}/{}/{} - {}/{}/{}".format(up_timer_2,center_timer_2,down_timer_2,left_timer_2,mid_timer_2,right_timer_2), (int(width*0.50), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)

                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    c1=int((p1[0]+p2[0])/2)
                    c2=int((p1[1]+p2[1])/2)
                    
                    X2.append(c1)
                    Y2.append(c2)

                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                    #track on black image
                    cv2.circle(img,(c1,c2),2,colors[i],-1)


                    #mid timer
                    if(c2>=int(vertical_mirror_line_left) and c1<=int(vertical_mirror_line_right)):
                       mid_timer_2=int(global_time-(left_timer_2+right_timer_2))   
                    #left timer
                    elif(c2<int(vertical_mirror_line_left)):
                        left_timer_2=int(global_time-(right_timer_1+mid_timer_1))   
                    #right timer
                    elif(c2>int(vertical_mirror_line_right)):
                        right_timer_2=int(global_time-(left_timer_1+mid_timer_1))

                        
                    #is fish mid the tank? how many times?
                    if(c2>=int(vertical_mirror_line_left) and c1<=int(vertical_mirror_line_right) and count_mid_2==True):
                        counter_mid_2+=1
                        count_mid_2=False
                        count_left_2=True
                        count_right_2=True
                        mid_counter_2=int(counter_mid_2)
                    #is fish left the tank? how many times?
                    elif(c2<int(vertical_mirror_line_left) and count_left_2==True):
                        counter_left_2+=1
                        count_left_2=False
                        count_mid_2=True
                        count_right_2=True
                        left_counter_2=int(counter_left_2)   
                    #is fish right the tank? how many times?
                    elif(c2>int(vertical_mirror_line_right) and count_right_2==True):
                        counter_right_2+=1
                        count_right_2=False
                        count_left_2=True
                        count_mid_2=True
                        right_counter_2=int(counter_right_2)
                        

                    #center timer
                    if(c2>=(int(half_height)-int(vertical_point)) and c2<=(int(half_height)+int(vertical_point))):
                        center_timer_2=int(global_time-(up_timer_2+down_timer_2))
                    #above timer
                    elif(c2<(int(half_height)-int(vertical_point))):
                        up_timer_2=int(global_time-(down_timer_2+center_timer_2))
                    #below timer
                    elif (c2>(int(half_height)+int(vertical_point))):
                        down_timer_2=int(global_time-(up_timer_2+center_timer_2))


                    #is fish center the tank? how many times?
                    if(c2>=(int(half_height)-int(vertical_point)) and c2<=(int(half_height)+int(vertical_point)) and count_center_2==True):
                        counter_center_2+=1
                        count_center_2=False
                        count_up_2=True
                        count_down_2=True
                        center_counter_2=int(counter_center_2)
                    #is fish above the tank? how many times?
                    elif(c2<(int(half_height)-int(vertical_point)) and count_up_2==True):
                        counter_up_2+=1
                        count_up_2=False
                        count_center_2=True
                        count_down_2=True
                        up_counter_2=int(counter_up_2)
                    #is fish below the tank? how many times?
                    elif(c2>(int(half_height)+int(vertical_point)) and count_down_2==True):
                        counter_down_2+=1
                        count_down_2=False
                        count_up_2=True
                        count_center_2=True
                        down_counter_2=int(counter_down_2)

                        
                    #append position
                    if current_time - start_time >= interval:
                        position_2.append(int(c1))
                        print(position_2)
                        if(selector==2):
                            start_time = current_time



                #IF FISH 3 EXISTS 
                if(i==2):
                    cv2.putText(frame, "F3C (TOP/CENTER/BOTTOM - LEFT/MID/RIGHT) = {}/{}/{} - {}/{}/{}".format(up_counter_3,center_counter_3,down_counter_3,left_counter_3,mid_counter_3,right_counter_3), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)
                    cv2.putText(frame, "F3T (TOP/CENTER/BOTTOM - LEFT/MID/RIGHT) = {}/{}/{} - {}/{}/{}".format(up_timer_3,center_timer_3,down_timer_3,left_timer_3,mid_timer_3,right_timer_3), (int(width*0.50), 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 0, cv2.LINE_AA)

                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    c1=int((p1[0]+p2[0])/2)
                    c2=int((p1[1]+p2[1])/2)
                    
                    X3.append(c1)
                    Y3.append(c2)

                    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
                    #track on black image
                    cv2.circle(img,(c1,c2),2,colors[i],-1)

                    
                    #mid timer
                    if(c2>=int(vertical_mirror_line_left) and c1<=int(vertical_mirror_line_right)):
                       mid_timer_3=int(global_time-(left_timer_3+right_timer_3))   
                    #left timer
                    elif(c2<int(vertical_mirror_line_left)):
                        left_timer_3=int(global_time-(right_timer_3+mid_timer_3))   
                    #right timer
                    elif(c2>int(vertical_mirror_line_right)):
                        right_timer_3=int(global_time-(left_timer_3+mid_timer_3))

                        
                    #is fish mid the tank? how many times?
                    if(c2>=int(vertical_mirror_line_left) and c1<=int(vertical_mirror_line_right) and count_mid_3==True):
                        counter_mid_3+=1
                        count_mid_3=False
                        count_left_3=True
                        count_right_3=True
                        mid_counter_3=int(counter_mid_3)
                    #is fish left the tank? how many times?
                    elif(c2<int(vertical_mirror_line_left) and count_left_3==True):
                        counter_left_3+=1
                        count_left_3=False
                        count_mid_3=True
                        count_right_3=True
                        left_counter_3=int(counter_left_3)   
                    #is fish right the tank? how many times?
                    elif(c2>int(vertical_mirror_line_right) and count_right_3==True):
                        counter_right_3+=1
                        count_right_3=False
                        count_left_3=True
                        count_mid_3=True
                        right_counter_3=int(counter_right_3)
                        

                    #center timer
                    if(c2>=(int(half_height)-int(vertical_point)) and c2<=(int(half_height)+int(vertical_point))):
                        center_timer_3=int(global_time-(up_timer_3+down_timer_3))
                    #above timer
                    elif(c2<(int(half_height)-int(vertical_point))):
                        up_timer_3=int(global_time-(down_timer_3+center_timer_3))
                    #below timer
                    elif (c2>(int(half_height)+int(vertical_point))):
                        down_timer_3=int(global_time-(up_timer_3+center_timer_3))


                    #is fish center the tank? how many times?
                    if(c2>=(int(half_height)-int(vertical_point)) and c2<=(int(half_height)+int(vertical_point)) and count_center_3==True):
                        counter_center_3+=1
                        count_center_3=False
                        count_up_3=True
                        count_down_3=True
                        center_counter_3=int(counter_center_3)

                    #is fish above the tank? how many times?
                    elif(c2<(int(half_height)-int(vertical_point)) and count_up_3==True):
                        counter_up_3+=1
                        count_up_3=False
                        count_center_3=True
                        count_down_3=True
                        up_counter_3=int(counter_up_3)

                    #is fish below the tank? how many times?
                    elif(c2>(int(half_height)+int(vertical_point)) and count_down_3==True):
                        counter_down_3+=1
                        count_down_3=False
                        count_up_3=True
                        count_center_3=True
                        down_counter_3=int(counter_down_3)

                    #append position
                    if current_time - start_time >= interval:
                        position_3.append(int(c1))
                        print(position_3)
                        if(selector==3):
                            start_time = current_time       


        except:
            pass
                    
        global_time=int(time.time()-starting_time)
        # show frame
        cv2.imshow('MultiTracker', frame)
        cv2.imshow("img",img)
    
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    if(fish==1):
        return X1,Y1,0,0,0,0,position_1,[],[]
    elif(fish==2):
        return X1,Y1,X2,Y2,0,0,position_1,position_2,[]
    else:
        return X1,Y1,X2,Y2,X3,Y3,position_1,position_2,position_3
        
    



def control():
    videoPath="control.mp4"
    cCap=cv2.VideoCapture(videoPath)
    cX1,cY1,cX2,cY2,cX3,cY3,cPosition_1,cPosition_2,cPosition_3=fish(cCap)
    response=input("retake? y/n")
    if(response=="y"):
        cCap.release()
        cv2.destroyAllWindows()
        control()
    else:
        return cX1,cY1,cX2,cY2,cX3,cY3,cPosition_1,cPosition_2,cPosition_3


def toxic():
    videoPath="toxin.mp4"
    tCap=cv2.VideoCapture(videoPath)
    tX1,tY1,tX2,tY2,tX3,tY3,tPosition_1,tPosition_2,tPosition_3=fish(tCap)
    response=input("retake? y/n")
    if(response=="y"):
        tCap.release()
        cv2.destroyAllWindows()
        toxic()
    else:
        return tX1,tY1,tX2,tY2,tX3,tY3,tPosition_1,tPosition_2,tPosition_3

            
#tracker types
trackerTypes=["CSRT","MIL"]

#control
cX1,cY1,cX2,cY2,cX3,cY3,cPosition_1,cPosition_2,cPosition_3=control()
cv2.destroyAllWindows()

#toxin
tX1,tY1,tX2,tY2,tX3,tY3,tPosition_1,tPosition_2,tPosition_3=toxic()

#graph plotting stuff
fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")

if(cX1!=0):
    ax.plot(cX1,cY1,label="Fish 1 (Control)")
if(cX2!=0):
    ax.plot(cX2,cY2,label="Fish 2 (Control)")
if(cX3!=0):
    ax.plot(cX3,cY3,label="Fish 3 (Control)")

if(tX1!=0):
    ax.plot(tX1,tY1,label="Fish 1 (Toxin)")
if(tX2!=0):
    ax.plot(tX2,tY2,label="Fish 2 (Toxin)")
if(tX3!=0):
    ax.plot(tX3,tY3,label="Fish 3 (Toxin)")

plt.legend(loc="upper left")
plt.show()

cP1=np.array(cPosition_1)
cP2=np.array(cPosition_2)
cP3=np.array(cPosition_3)

tP1=np.array(tPosition_1)
tP2=np.array(tPosition_2)
tP3=np.array(tPosition_3)

if(cP1.size>0):    
    plt.plot(cP1,label="Fish 1 (Control)")
if(cP2.size>0):
    plt.plot(cP2,label="Fish 2 (Control)")
if(cP3.size>0):
    plt.plot(cP3,label="Fish 3 (Control)")
    
if(tP1.size>0):
    plt.plot(tP1,label="Fish 1 (Toxin)")
if(tP2.size>0):
    plt.plot(tP2,label="Fish 2 (Toxin)")
if(tP3.size>0):
    plt.plot(tP3,label="Fish 3 (Toxin)")

plt.legend(loc="upper left")
plt.show()

excelUpdater(cPosition_1,cPosition_2,cPosition_3,tPosition_1,tPosition_2,tPosition_3)
