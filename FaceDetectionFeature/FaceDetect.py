#Stage1: Import all of the installed dependencies
import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime
from imutils.video import VideoStream


#Stage2: Identify the path of images used for training
path='ImagesAttendance'
perNames=[]
imgNames=[]
studentList=os.listdir(path)

#Stage3: Encodings for face detection
def applyEncodings(imgNames):
    encodingList=[]
    for pics in imgNames:
        pics=cv2.cvtColor(pics,cv2.COLOR_BGR2RGB)
        encodingProcess=face_recognition.face_encodings(pics)[0]
        encodingList.append(encodingProcess)

    return encodingList
    knownList = applyEncodings(imgNames)

#Stage4: Attendance Marking System
def takeAttendance(studentNames):
    with open('Attendance.csv','r+') as f:
        trackData = f.readlines()
        snameList=[]
        for section in trackData:
            division= section.split(',')
            snameList.append(division[0])
        if name not in snameList:
            current = datetime.current()
            entryTime = current.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{entryTime}')


        print(trackData)

#Stage5: Find student names from the images given
for pNames in studentList:
    curPic=cv2.imread(f'{path}/{pNames}')
    imgNames.append(curPic)
    perNames.append(os.path.splitext(pNames)[0])

#Stage6: Implementation of demo video
encodeListKnown = applyEncodings(imgNames)
cap = VideoStream('ZoomDemoVid.mp4').start()

#Stage7: Convert imgs to rgb and recognize face
while True:
    pics= cap.read()
    rgb_pics = pics[:, :, ::-1]

    smallPics= cv2.resize(pics,(0, 0), None, 0.25, 0.25)
    smallPics=cv2.cvtColor(pics, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_pics)
    faceFrame = face_recognition.face_locations(smallPics)
    encodingFrame= face_recognition.face_encodings(smallPics,faceFrame)

#Stage8: Add box around the faces
    for faceEncode,faceLoc in zip(encodingFrame,faceFrame):
        faceMatch= face_recognition.compare_faces(encodeListKnown,faceEncode)
        faceDis=face_recognition.face_distance(encodeListKnown,faceEncode)
        matchIndicator= np.argmin(faceDis)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(pics, (left, top), (right, bottom), (0, 0,255), 2)
            cv2.imshow('FaceDetect', pics)

#Stage9: Match and attendance feature
        if faceMatch[matchIndicator]:
            name = perNames[matchIndicator].upper()
            print(name)
            takeAttendance(name)
            cv2.waitKey(1)