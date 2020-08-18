#Stage1: Import all of the installed dependencies
import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime
from imutils.video import VideoStream
import pandas as pd

#Stage2: Identify the path of images used for training
path='/content/ImagesAttendance'
perNames=[]
imgNames=[]
studentList=os.listdir(path)

identified_names = [] # Empty list to store names of identified people
arrival_record = [] # Empty list to store names, time of first detection, punctuality

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
def takeAttendance(studentNames,timestamp):
      if name not in identified_names:
          identified_names.append(name)
          timestamp_in_secs = int(timestamp/1000)
          secs = int(timestamp_in_secs%60)
          mins = int(timestamp_in_secs/60)
            # Set status to late if they arrive after 5 mins
          if mins>=5 and secs>0:
              status = 'Late'
          else:
              status = 'On time'
          if secs < 10:
            arrival_record.append((name,f'{str(mins)}:0{str(secs)}',status))
          else:
            arrival_record.append((name,f'{str(mins)}:{str(secs)}',status))



#Stage5: Find student names from the images given
for pNames in studentList:
    curPic=cv2.imread(f'{path}/{pNames}')
    imgNames.append(curPic)
    perNames.append(os.path.splitext(pNames)[0])

#Stage6: Implementation of demo video
encodeListKnown = applyEncodings(imgNames)
cap = cv2.VideoCapture('ZoomDemoVid.mp4')

#Stage7: Create writer to write output mp4 file with red boxes showing face detection
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter('demoOutput.mp4',fourcc, 20,(1280, 720), True)


#Stage8: Convert imgs to rgb and recognize face
while (cap.isOpened()):
    ret, pics= cap.read()
    if ret == True:
      timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
      rgb_pics = pics[:, :, ::-1]

      smallPics= cv2.resize(pics,(0, 0), None, 0.25, 0.25)
      smallPics=cv2.cvtColor(pics, cv2.COLOR_BGR2RGB)

      face_locations = face_recognition.face_locations(rgb_pics) #boxes
      faceFrame = face_recognition.face_locations(smallPics)
      encodingFrame= face_recognition.face_encodings(smallPics,faceFrame)

  #Stage9: Add box around the faces
      for faceEncode,faceLoc in zip(encodingFrame,faceFrame):
          faceMatch= face_recognition.compare_faces(encodeListKnown,faceEncode)
          faceDis=face_recognition.face_distance(encodeListKnown,faceEncode)
          matchIndicator= np.argmin(faceDis)

          for top, right, bottom, left in face_locations:
              cv2.rectangle(pics, (left, top), (right, bottom), (0, 0,255), 2)

  #Stage10: Match and attendance feature
          if faceMatch[matchIndicator]:
              name = perNames[matchIndicator].upper()
              print(f"[ANALYSING AND RENDERING VIDEO AT {int(timestamp/1000)} SECONDS...]")
              takeAttendance(name,timestamp)
              cv2.waitKey(1)
      if writer is not None:
		      writer.write(pics)
    else:
        break

#Stage11: Clean up and export mp4 video
cv2.destroyAllWindows()
writer.release()

#Stage12: Create Pandas dataframe to store attendance record and then output as CSV file
names = []
arrival_time = []
punctuality = []

for name, time, status in arrival_record:
  names.append(name)
  arrival_time.append(time)
  punctuality.append(status)

output = pd.DataFrame()
output["Student Name"] = names
output["Arrival Time"] = arrival_time
output["Punctuality"] = punctuality

output.to_csv("AttendanceReport.csv", index=False)