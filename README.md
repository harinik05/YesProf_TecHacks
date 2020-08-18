## YesProf! : Novel-Based Approach to Facilitate Virtual Learning

This technology consists of two significant entities that include attendence tracking feature using face recognition and pdf-generated engagement session report. It purely fulfills its purpose of providing student data such as attendance and reports to instructors to track their students' progress and drive them in the correct path of achievement. 

### Stage 1: Attendance Tracker 

Once entering into the virtual meeting environment(such as zoom or google meet), this feature enables face recognition of students to track their presence and punctuality. This feature was built using OpenCV and python. The dependencies that were used are listed below:
1. cv2
2. numpy np
3. os
4. face_recognition
5. datetime
6. imutils VideoStream

Once these dependencies are installed using pycharm file settings, they can be imported using the following commands:
```
import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime
from imutils.video import VideoStream
```
Save the images from ImagesAttendance File and use them for the path. Find encodings and mark attendance to connect directly to the csv file (which can be opened as an excel document). 

The boxes are added around the faces and this will be run through the demo video (mp4 file format). This will allow the recognition of faces which is recorded in the tracker for teachers to utilize.


### Stage 2: Session Engagement Reports
After the virtual meeting is over, chat and conversation history is analysed to produced Engagement reports for each participant based on their contributions. This includes a PDF file which contains all the data visualizations and a CSV file containing the report summary. This feature was built using Python, SpeechRecognition and Matplotlib. The dependencies that were used are listed below:
1. speech_recognition
2. pyaudio
3. os
4. pydub AudioSegment
5. pydub.silence split_on_silence
6. wave
7. contextlib
8. pandas as pd
9. numpy as np
10. matplotlib.pyplot as plt
11. matplotlib.backends.backend_pdf import PdfPages
12. datetime
13. argparse

Once these dependencies are installed, they can be imported using the following commands:
```
import speech_recognition as sr
import pyaudio
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import wave
import contextlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import argparse

```

To produce a PDF and CSV Engagement Report on a demo, run the EngagementAnalysis.py script by passing command line arguments for paths to the example chat and audio file (default settings to files in ZoomDemo folder).  You can also use the command line argument -t to set a custom PDF title. This will then export a PDF and CSV file into the same directory as the python script. To avoid a RuntimeWarning in your command prompt, you can try running the main script as 'python -W EngagementAnalysis.py'.

There is also a walk-through Jupyter Notebook available in the same folder to demonstrate how the report was produced and contains further clarifications on methods used. All files exported from this notebook are stored in the GraphReports folder. Note that this feature should run real-time speech engagement analysis in an ideal application (through video conferencing platforms), thus speaker classification was hard-coded in this project for demonstration purposes only.
