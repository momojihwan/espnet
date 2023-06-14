# -*- coding: utf-8 -*-
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time, datetime
import subprocess
import scipy.io as sio
import scipy.io.wavfile
from scipy import signal
import sounddevice as sd
import pyaudio
import sys, os, time
import matplotlib
import matplotlib.pyplot as plt
 
matplotlib.use('Qt5Agg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from queue import PriorityQueue
# triton client
from tritonclient.utils import *
import tritonclient.http as httpclient
import torchaudio
# from models.E2E import E2E
from audio_spectrum import AudioStream, Display_status
import numpy as np
import wave

import ASR_main

CHUNK = 1024             # samples per frame
WAVDIR = "/home/mozi/Workspace/espnet/egs2/librispeech_100/asr1/data/test_clean/wav.scp"
TEXTDIR =  "/home/mozi/Workspace/espnet/egs2/librispeech_100/asr1/data/test_clean/text"

INF = 123456789

class ASRDialog(QMainWindow, ASR_main.Ui_KT):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

        self.wavFilePath = ""
        self.wavFileName = ""
        self.textFilePath = ""
        self.textFileName = ""

        self.sample_rate = None
        self.data = None
    
        self.display_status = Display_status(self)
        # self.display_status.start()
        self.display_status.finished.connect(self.display_pred_script)

        self.plot_wavform = AudioStream(self)
        self.plot_wavform.streaming_start.connect(self.streaming_decoding)

        self.label_widget.setAlignment(QtCore.Qt.AlignCenter)
        self.file_list = {}
        self.script_list = {}
        self.View_List = []
        self.LoadFileList()
        self.file_index = self.View_List
        self.FileList.setStyleSheet("color: white;")
        self.FileList.setCurrentIndex(0)
        self.FileList.currentIndexChanged['QString'].connect(self.update_now)

        # Button Event
        self.play_button.clicked.connect(self.btnPlayClicked)
        self.play_button.pressed.connect(self.btnPlayPressed)
        self.play_button.released.connect(self.btnPlayReleased)
        
        self.stop_button.clicked.connect(self.btnStopClicked)
        self.stop_button.pressed.connect(self.btnStopPressed)
        self.stop_button.released.connect(self.btnStopReleased)


        # self.canvas = CanvasWidget()
        self.canvas = self.plot_wavform.canvas
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.audio_widget.setLayout(layout)

    # Button Press Event
    def btnPlayPressed(self):
        self.play_button.setStyleSheet("image: url(./Pictures/play.png);\nborder: 0px;")
    def btnStopPressed(self):
        self.stop_button.setStyleSheet("image: url(./Pictures/stop.png);\nborder: 0px;")

    # Button Release Event
    def btnPlayReleased(self):
        self.play_button.setStyleSheet("image: url(./Pictures/play_clicked.png);\nborder: 0px;")
    def btnStopReleased(self):
        self.stop_button.setStyleSheet("image: url(./Pictures/stop_clicked.png);\nborder: 0px;")
    
    # Button Click Event
    def btnPlayClicked(self):           # Play 버튼 클릭 시 

        # Display Audio stream
        self.plot_wavform.pause = False
        self.plot_wavform.wavFilePath = self.wavFilePath
        self.plot_wavform.start()
        # self.plot_wavform.one_wav_start_plot(self.wavFilePath)
        self.canvas.axs[0].cla()
    
    def streaming_decoding(self, idx):
        self.display_status.idx = idx
        self.display_status.start()

    def display_pred_script(self, object):
        text = object.astype(str)
        
        log_print = '{}'.format(text)
        self.pred_widget.setText(log_print)

    def display_label_script(self):
        self.label_widget.setText(self.script_list[self.wavFileName])

    def btnStopClicked(self):           # Stop 버튼 클릭 시
        sd.stop()
        self.plot_wavform.pause = True

    def LoadFileList(self):
        
        with open(WAVDIR) as f:
            lines = f.readlines()
            for line in lines:
                wav_id, wav_path = line.split()
                self.file_list[wav_id] = wav_path
                self.View_List.append(wav_id)
        
        with open(TEXTDIR) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                wav_id = line[0]
                wav_script = " ".join(line[1:])
                self.script_list[wav_id] = wav_script

        self.FileList.addItems(self.View_List)
        self.wavFileName = self.View_List[0]
        self.wavFilePath = self.file_list[self.wavFileName]
        self.display_label_script()
        # self.sample_rate, self.data = sio.wavfile.read(self.wavFilePath)

        #subprocess.call('docker exec KT-docker /home/Workspace/HYnet/egs/ksponspeech/asr1/local/feature_extraction.sh', shell=True)

    def get_text(self, text):

        textfile_path = PREDDIR + '/' + text
        textFile = open(textfile_path, 'r')
        line = textFile.readline()
        textline = line.rstrip('\n')
        return textline

    def update_now(self, value):
        self.file_index = self.View_List.index(value)
        self.wavFileName = value
        self.wavFilePath = self.file_list[self.wavFileName]
        self.display_label_script()
        # self.Display_Wavform(self.file_index)
     
    def Display_Wavform(self, index):           
        
        self.canvas.axs[0].cla()
        
        item = self.Model.itemFromIndex(index)

        self.wavFileName = item.text()
        self.wavFilePath = WAVDIR + "/" + self.wavFileName
        self.FileName_label.setText(self.wavFileName)

        self.sample_rate, self.data = sio.wavfile.read(self.wavFilePath)
        
        # self.canvas.update_(self.data)
        

app = QApplication(sys.argv)
dlg = ASRDialog()
dlg.show()
app.exec_()