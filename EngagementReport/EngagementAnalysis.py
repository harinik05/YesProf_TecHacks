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

# Set default for path files to chat and audio input
parser = argparse.ArgumentParser()
parser.add_argument('--chat', default="../ZoomDemo/meeting_saved_chat.txt", dest="chat",help="path to input chat .txt file")
parser.add_argument('--audio', default="../ZoomDemo/ZoomDemoAudio.wav", dest="audio", help="path to input audio .wav file")
parser.add_argument('-t', default='Engagement Analysis for Zoom', dest="pdf_title", help="title of exported pdf summary")
args = parser.parse_args()


chat_path = args.chat
with open(chat_path, 'rt') as fd:
        contents = fd.readlines()
        
# Dictionary to story texts per person and total words per person
texts_person = {}
words_per_person = {}

# Tuple unpacking for file_name, contents
for line in contents:
    line_no_time = line.split('\t')[1]
    person, text = line_no_time.split(':',1)
    person = " ".join(person.split()[1:])
    if person in texts_person:
        texts_person[person].append(line)
        words_per_person[person] += len(text)
    else:
        texts_person[person] = [line]
        words_per_person[person] = len(text.split())

#  Function that splits the audio file into chunks and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    r = sr.Recognizer()
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = []
    current_time = 0
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with contextlib.closing(wave.open(chunk_filename,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            current_time += duration
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                pass
            else:
                text = f"{text.capitalize()}. "
                pass
                whole_text.append((round(current_time,2),text))
    # return the text for all chunks detected
    return whole_text

audio_path = args.audio
transcript = get_large_audio_transcription(audio_path)

# Tuple unpacking on transcript as time added for debugging purposes
lines = [text for _,text in transcript]

# Hard-coding classification of speaker for each line of transcript (only for demo purposes)
people = list(words_per_person.keys())
# Encode each line by index of person speaking in list above
person_line_keys = [2,0,0,0,0,0,4,0,0,2,0,0,2,3,0,0,2,0,1,2,2,2,1,0,0,1,0,0,3,2,2,2,3,0,2,3,0,4,0,4,4,4,0,2,0,2,2,2,0,0,0,2,2,0,2,2,2,2,0,0,2,0,0,0,0,3,0,0,2,0,0,2,0,2,2,0,0,3,2,0,0,0,2,0,0,0,0,0]

# Create empty dict to track words spoken by each person
words_spoken_per_person = {}
for i, index in enumerate(person_line_keys):
    person = people[index]
    num_words = len(lines[i])
    if person in words_spoken_per_person.keys():
        words_spoken_per_person[person] += num_words
    else:
        words_spoken_per_person[person] = num_words

# Find max length of lines of text per person (to avoid error during csv conversion)
texts_list = list(texts_person.values())
max_length = 0
for person in texts_list:
    max_length = max(max_length,len(person))
    
# Producing table as a summary report
spoken_summary = [(person,words_spoken_per_person[person]) for person in words_spoken_per_person.keys()]

# Sort by words spoken
spoken_summary.sort(key=lambda x:x[1])
spoken_summary = spoken_summary[::-1]

# Replace old csv with one that includes words spoken per person
output = pd.DataFrame()
for name,spoken_words in spoken_summary:
    contents = []
    word_num = words_per_person[name]
    
    if word_num == 1:
        contents.append(str(spoken_words)+" word spoken") 
    else:
        contents.append(str(spoken_words)+" words spoken") 
        
    if word_num == 1:
        contents.append(str(words_per_person[name])+" word typed") 
    else:
        contents.append(str(words_per_person[name])+" words typed") 
        
    contents.append("Zoom chat contributions:")
    for line in texts_person[name]:
        contents.append(line)
        
    # Add empty lines to shorter columns
    empty_lines = max_length - len(texts_person[name])
    for line in range(empty_lines):
        contents.append("")
        
    output[name] = contents

# Output as CSV file
output.to_csv("chatEngagementSummary.csv", index=False)

with PdfPages('EngagementAnalysis.pdf') as export_pdf:
    
    # Plotting a pie chart in term of engagement by text
    plt.axis("equal")
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','indianred']
    plt.pie([float(v) for v in words_spoken_per_person.values()], labels=[k for k in words_spoken_per_person.keys()],autopct='%1.1f%%', colors=colors, shadow=True,  startangle=140)
    plt.title('Speech engagement as percentage of total words in Zoom conversation')
    export_pdf.savefig(bbox_inches = 'tight',papertype = 'a4', orientation = 'portrait')
    plt.close()
    
    # Plotting a bar chart in term of engagement by text
    colors = ["#ffc130","#fa5f8b","#73c799","#6adeeb","#d77df0"]
    plt.style.use('ggplot')
    plt.barh(width=[float(v) for v in words_spoken_per_person.values()], y=[k for k in words_spoken_per_person.keys()], color=colors, height =0.5)
    plt.ylabel("Student")
    plt.xlabel("Words Spoken")
    plt.title('Engagement in terms of words spoken during Zoom session')
    export_pdf.savefig(bbox_inches = 'tight')
    plt.close()
    
    # Plotting a pie chart in term of engagement by speech
    plt.axis("equal")
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','indianred']
    plt.pie([float(v) for v in words_spoken_per_person.values()], labels=[k for k in words_spoken_per_person.keys()],autopct='%1.1f%%', colors=colors, shadow=True,  startangle=140)
    plt.title('Speech engagement as percentage of total words in Zoom conversation')
    export_pdf.savefig(bbox_inches = 'tight')
    plt.close()
    
    # Plotting a bar chart in term of engagement by speech
    colors = ["#ffc130","#fa5f8b","#73c799","#6adeeb","#d77df0"]
    plt.style.use('ggplot')
    plt.barh(width=[float(v) for v in words_spoken_per_person.values()], y=[k for k in words_spoken_per_person.keys()], color=colors, height =0.5)
    plt.ylabel("Student")
    plt.xlabel("Words Spoken")
    plt.title('Engagement in terms of words spoken during Zoom session')
    export_pdf.savefig(bbox_inches = 'tight')
    plt.close()
    
    # Set the file's metadata via the PdfPages object:
    d = export_pdf.infodict()
    d['Title'] = args.pdf_title
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['CreationDate'] = datetime.datetime(2020, 8, 16)
    d['ModDate'] = datetime.datetime.today()