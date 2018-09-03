# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages

import os
import cv2
import time
import math
import imutils
import argparse
import geocoder
import webbrowser
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from imutils.video import FPS
from imutils.video import VideoStream
from tkinter import *

def color(img,gray,c):
    c_red = [[0, 25, 170], [10, 80, 220]]
    c_blue = [[150, 25, 0], [220, 85, 10]]
    c_green = [[0, 170, 0], [10,255,10]]
    mask = np.zeros(gray.shape,np.uint8)
    cv2.drawContours(mask,[c],0,255,-1)
    mean = cv2.mean(img,mask=mask)
    mean = list(map(int,mean))
    check = 0
    b = mean[0]
    g = mean[1]
    r = mean[2]
    if(b<50 and g<50 and r<50):
        return "black"
    for i in range(0,3):
        if mean[i]>=c_red[0][i] and mean[i]<=c_red[1][i]:
            check += 1
        else:
            if mean[i] < c_red[0][i]:
                if abs(mean[i] - c_red[0][i]) < 10:
                    check += 1
                else:
                    break
            elif mean[i] > c_red[1][i]:
                if abs(mean[i] - c_red[1][i]) < 10:
                    check += 1
                else:
                    break
    if check==3:
        return "red"
    check = 0
    for i in range(0,3):
        if mean[i]>=c_blue[0][i] and mean[i]<=c_blue[1][i]:
            check += 1
        else:
            if mean[i] < c_blue[0][i]:
                if abs(mean[i] - c_blue[0][i]) < 10:
                    check += 1
                else:
                    break
            elif mean[i] > c_blue[1][i]:
                if abs(mean[i] - c_blue[1][i]) < 10:
                    check += 1
                else:
                    break
    if check==3:
        return "blue"

    check = 0
    for i in range(0,3):
        if mean[i]>=c_green[0][i] and mean[i]<=c_green[1][i]:
            check += 1
        else:
            if mean[i] < c_green[0][i]:
                if abs(mean[i] - c_green[0][i]) < 10:
                    check += 1
                else:
                    break
            elif mean[i] > c_green[1][i]:
                if abs(mean[i] - c_green[1][i]) < 10:
                    check += 1
                else:
                    break
                
    if check==3:
        return "green"

    b = 255 - mean[0]
    g = 255 - mean[1]
    r = 255 - mean[2]
    c = [(b,'b'),(g,'g'),(r,'r')]
    c = min(c)
    if c[1]=='b':
        return "blue"
    if c[1] == 'g':
        return "green"
    if c[1] == 'r':
        return "red"
    return "black"



def shape(c):
    # initialize the shape name and approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    size = cv2.contourArea(c)
    if len(approx) == 3:
        return ["Triangle",size]
    elif len(approx) == 4:
        return ["4-sided",size]
    elif len(approx) == 5:
        return ["5-sided",size]
    else:
        return ["Circle",size]


    
def getobjects(source):
    img = cv2.imread(source)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(gray,kernel,iterations=1)
    blur = cv2.GaussianBlur(dilation,(5,5),0)
    ret,thresh = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)
    grid_size = (img.shape[0]/10)*(img.shape[1]/10)
    grid_size = grid_size - ((grid_size*10)/100) #10% is removed
    objects = {}
    for j in range(1,11):
        for i in range(1,11):
            temp = thresh[int((i-1)*(img.shape[0]/10)):int(i*(img.shape[0]/10)),int((j-1)*(img.shape[1]/10)):int(j*(img.shape[1]/10))]
            check = np.count_nonzero(temp)
            if check < grid_size:
                objects[(j,i)]= []
    _,contours,h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        cv2.drawContours(img,contours,i,(0,255,0),2)
        M = cv2.moments(contours[i])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        for k in objects:
            if cx > (k[0]-1)*(img.shape[0]/10) and cx < (k[0])*(img.shape[0]/10):
                if cy > (k[1]-1)*(img.shape[0]/10) and cy < (k[1])*(img.shape[0]/10):
                    s = shape(contours[i])
                    objects[k]=([color(img,gray,contours[i]),s[0],s[1]])
    return objects



def isexists(objects,source):
    if source in objects.keys():
        elements = objects[source]
    else:
        return 0
    if elements[0] == "black":
        return 0
    for i in objects:
        if i != source:
            if elements[:2] == objects[i][:2] and abs(elements[2] - objects[i][2]) < 3:
                return 1
    return 0



def areequal(source,dest,objects):
    if (dest in objects) and source!=dest:
        if (objects[dest][:2] == objects[source][:2]) and abs(objects[dest][2] - objects[source][2]) < 3:
            return 1
    return 0



def neighbors(current,objects):
    up = (current[0]-1,current[1])
    right = (current[0],current[1]+1)
    down = (current[0]+1,current[1])
    left = (current[0],current[1]-1)
    n = []
    dest = up
    if not (dest[0] < 1 or dest[0] > 10 or dest[1] > 10 or dest[1] < 1 or ((dest in objects) and (objects[dest][0]=="black"))):
        n.append(dest)
    dest = right
    if not (dest[0] < 1 or dest[0] > 10 or dest[1] > 10 or dest[1] < 1 or ((dest in objects) and (objects[dest][0]=="black"))):
        n.append(dest)
    dest = down
    if not (dest[0] < 1 or dest[0] > 10 or dest[1] > 10 or dest[1] < 1 or ((dest in objects) and (objects[dest][0]=="black"))):
        n.append(dest)
    dest = left
    if not (dest[0] < 1 or dest[0] > 10 or dest[1] > 10 or dest[1] < 1 or ((dest in objects) and (objects[dest][0]=="black"))):
        n.append(dest)
    return n



def getDir(source,dest):
    if(source[0]==dest[0]):
        if source[1]<dest[1]:
            return "down"
        else:
            return "up"
    elif(source[1]==dest[1]):
        if source[0]<dest[0]:
            return "right"
        else:
            return "left"



def printDirections(path):
    res = []
    res.append(path[0])
    for i in range(1,len(path)):
        res.append(getDir(path[i-1],path[i]))
    res.append(path[-1])
    print (res)



def findpath(objects,source,dest):
    frontier = list()
    frontier.append(source)
    came_from = {}
    came_from[source] = None

    while len(frontier) > 0:
       current = frontier.pop(0)
       if(current==dest):
           break
       for next in neighbors(current,objects):
          if next not in came_from:
             if next in objects:
                 if next==dest:
                    frontier.append(next)
                    came_from[next] = current
             else:
                 frontier.append(next)
                 came_from[next] = current

    current = dest 
    path = [current]
    while current != source:
       current = came_from[current]
       path.append(current)
    path.append(source) # optional
    path.reverse() # optional
    print (path[1:])
    printDirections(path[1:])


if __name__ == '__main__':
    #creating instance of TK
    root=Tk()
    #Providing Back ground colour for the Frame
    root.configure(background="#80D8FF")

	#Function to be performed when "Shortest distance between any 2 objects" button is pressed
    def function1():    
        Tk().withdraw()
        source=askopenfilename()
        if((source.lower().endswith('.jpg')) or (source.lower().endswith('.png'))):
                    #Tk().withdraw()
                    #source=askopenfilename()
                    #source = "test_image4.jpg"
                    objects = getobjects(source)
                    for i in sorted(objects):
                        if(objects[i][0]!='black'):
                            print (i, objects[i])
                    print ("Select two objects: x1 y1 , x2 y2")
                    x1, y1 = map(int,input().split(' '))
                    x2, y2 = map(int, input().split(' '))
                    if((x1,y1) not in objects):
                        print ((x1,y1),"not listed in objects")
                        exit(1)
                    if ((x2, y2) not in objects):
                        print ((x2, y2), "not listed in objects")
                        exit(1)
                    print ("\n\nShortest Path: ")        
                    findpath(objects,(x1,y1),(x2,y2))
        else:
            print("Invalid file. Only Images are allowed")
        print("==============================================================================")

	    #Function to be performed when "Find similar object" button is pressed
    def function2():
        #exec(compile(open("func2.py", "rb").read(),"func2.py",'exec'))
        Tk().withdraw()
        source=askopenfilename()
        if((source.lower().endswith('.jpg')) or (source.lower().endswith('.png'))):
            objects = getobjects(source)
            z=sorted(objects.items(), key=lambda t: t[::-1])
            a = [x for x in z if x[1][0] != 'black']
            objects = getobjects(source)
            for i in sorted(objects):
                if(objects[i][0]!='black'):
                    print (i, objects[i])
            print ("Select object: x1 y1 ")
            x1, y1 = map(int,input().split(' '))
            x=0
            for i in a:
                if (i[0]!=(x1, y1)):
                    x=x+1
                else:
                    y=i[1]
                    break
            indx = a.index(((x1, y1),y))
            if(indx>0):
                z=indx-1
                for z in a:
                    if(abs(z[1][2]-y[2])<5):
                        print(z)
            print("============================================================================")
        
	#Function to be performed when "Real time object detection" button is pressed
    #def function3():
     #   os.system("start cmd")
	#Function to train the data when "Shortest distance on map" button is pressed
    def function4():
        os.system("python func4.py")
        
	#setting title for the window
    root.title("SPD USING IP")
	#creating a text label
    Label(root, text="SELECT YOUR OPTION",font=("helvatica",40),fg="black",bg="#FFFFFF",height=2).grid(row=0,rowspan=2,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

	#creating a button
    Button(root,text="Shortest distance between any 2 objects",font=("times new roman",30),bg="#3F51B5",fg='white',command=function1).grid(row=3,columnspan=2,sticky=W+E+N+S,padx=5,pady=5)
	#creating second button
    Button(root,text="Find similar object",font=("times new roman",30),bg="#3F51B5",fg='white',command=function2).grid(row=4,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)

	#creating third button
    #Button(root,text="Real time object detection",font=('times new roman',30),bg="#3F51B5",fg="white",command=function3).grid(row=5,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)
	#creating Fourth button
    Button(root,text="Shortest distance on map",font=('times new roman',30),bg="#3F51B5",fg="white",command=function4).grid(row=7,columnspan=2,sticky=N+E+W+S,padx=5,pady=5)
    root.mainloop()
