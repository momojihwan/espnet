import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec
import wave
import pydub, array
from pydub import AudioSegment
import soundfile as sf
import struct
import scipy
from scipy.fftpack import fft
import sys
import time

from io import BytesIO

# triton client
from tritonclient.utils import *
import tritonclient.http as httpclient
import torchaudio, torch

CHUNK = 2048

def cutoffaxes(ax):  # facecolor='#000000'
    #ax.patch.set_facecolor(facecolor)

    ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

class CanvasWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=50):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='none')
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.setStyleSheet("background-color:transparent;")

        # buffer
        self.img_buffer = []

        # Initialize axis and lines
        gs = gridspec.GridSpec(1, 1)
        ax1 = self.fig.add_subplot(gs[:, :])
        #cutoffaxes(ax1, 'darkblue')
        cutoffaxes(ax1)
        self.axs = [ax1]
        self.fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        FigureCanvas.updateGeometry(self)

        # Compute init image to figure
        self._init_lines()


    def _init_lines(self, img=None):
        
        if img is None:
            x = np.arange(0, 2 * CHUNK, 2)

            line1, = self.axs[0].plot(x, np.random.rand(CHUNK), '-', lw=2, color='dimgray')

            # format waveform axes
            self.axs[0].set_title('AUDIO WAVEFORM')
            self.axs[0].set_xlabel('samples')
            self.axs[0].set_ylabel('volume')
            self.axs[0].set_ylim(-255, 511)
            self.axs[0].set_xlim(0, CHUNK)
            plt.setp(
            self.axs[0], yticks=[0, 128, 255],
            xticks=[0, CHUNK, CHUNK],
            )
            self.axs[0].set_facecolor('lightgray')
            # line1, = self.axs[0].plot([]) #, animated=True, cmap='gray'
        else:
            self.axs[0].grid(which='both')
            self.axs[0].set_ylim([min(img), max(img)])
            line1, = self.axs[0].plot(np.zeros(np.shape(img))) #, animated=True, cmap='gray'
        self.lines = [line1]

    def update_(self, img):
        self._init_lines(img)
        self.lines[0].set_xdata(range(len(img)))
        self.lines[0].set_ydata(img)
        self.draw()

    def stream_(self, data):
 
        self._init_lines()
        self.lines[0].set_xdata(range(len(data)))
        self.lines[0].set_ydata(data)
        self.draw()
        self.flush_events()

class Display_status(QThread):
    finished = pyqtSignal(object)
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.idx = None
        self.start_time = 0
        self.partial_audio_path = "/home/mozi/a.flac"
        self.model_name = "espnet"
        
    def run(self):
        wavfile = self.parent.wavFilePath
        # with httpclient.InferenceServerClient("localhost:8000") as client:
        with httpclient.InferenceServerClient("localhost:8000") as client:
            
            k = AudioSegment.from_raw(BytesIO(self.idx), sample_width=2, frame_rate=16000, channels=1).export(self.partial_audio_path, format="flac")
            w, sr = torchaudio.load(self.partial_audio_path)
            input_data = w.numpy()
            inputs = [
                httpclient.InferInput("INPUT", input_data.shape,
                                    np_to_triton_dtype(input_data.dtype)),
            ]

            inputs[0].set_data_from_numpy(input_data)

            outputs = [
                httpclient.InferRequestedOutput("OUTPUT")
            ]
            response = client.infer(self.model_name,
                                    inputs,
                                    request_id=str(1),
                                    outputs=outputs)
            
            result = response.get_response()
            output_data = response.as_numpy("OUTPUT")
            
            self.finished.emit(output_data)

    def end_log(self, text):

        end_time = time.time()
        log_time = datetime.datetime.now()

        log_print = '[{}] {}'.format(log_time, text)

        return log_print

class AudioStream(QThread):
    finished = pyqtSignal(bytes)
    streaming_start = pyqtSignal(bytes)
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.wavFilePath = ""

        self.partial_audio_path = "/home/mozi/a.flac"
        self.canvas = CanvasWidget()

        self.finished.connect(self.update_label_text)

        # Stream Constants
        self.CHUNK = CHUNK
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1 
        self.RATE = 16000
        self.pause = False

        self.p = pyaudio.PyAudio()
        
        # For Streaming
        # self.stream = self.p.open(
        #     format=self.FORMAT,
        #     channels=self.CHANNELS,
        #     rate=self.RATE,
        #     input=True,
        #     output=True,
        #     frames_per_buffer=self.CHUNK
        # )

    def update_label_text(self, idx):
        self.streaming_start.emit(idx)

    def run(self):
        wavFile = self.wavFilePath
        audio_segment = AudioSegment.from_file(wavFile, format='flac')
        raw_audio_data = audio_segment.raw_data

        # b = BytesIO(raw_audio_data)
        
        stream = self.p.open(
            format=self.p.get_format_from_width(audio_segment.sample_width),
            channels=audio_segment.channels,
            rate=audio_segment.frame_rate,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK
        )

        while not self.pause:
            
            for i in range(0, len(raw_audio_data), self.CHUNK):
                
                partial_audio = raw_audio_data[i:i+self.CHUNK]
                
                if len(partial_audio) != (self.CHUNK):
                    stream.write(raw_audio_data[i:])
                    self.finished.emit(raw_audio_data[i:])
                    self.canvas.axs[0].cla()
                    self.canvas.stream_(data_np)
                    break
                
                stream.write(partial_audio)
                data_int = struct.unpack(str(self.CHUNK) + 'B', partial_audio)
                data_np = np.array(data_int, dtype='b') + 128

                # k = AudioSegment.from_raw(BytesIO(raw_audio_data[:i+self.CHUNK]), sample_width=2, frame_rate=16000, channels=1).export(self.partial_audio_path, format="flac")
                self.finished.emit(raw_audio_data[:i+self.CHUNK])
                self.canvas.axs[0].cla()
                self.canvas.stream_(data_np)
                
                
            if len(partial_audio) != (self.CHUNK):
                    break
            
        stream.stop_stream()
        stream.close()
        

    # For Streaming        
    # def start_plot(self):

    #     frame_count = 0
    #     start_time = time.time()

    #     while not self.pause:
    #         data = self.stream.read(self.CHUNK)
    #         data_int = struct.unpack(str(self.CHUNK) + 'B', data)
    #         data_np = np.array(data_int, dtype='b')[::2] + 128

    #         self.line.set_ydata(data_np)

    #         # compute FFT and update line
    #         yf = fft(data_int)
    #         self.line_fft.set_ydata(
    #             np.abs(yf[0:self.CHUNK]) / (128 * self.CHUNK))

    #         # update figure canvas
    #         self.fig.canvas.draw()
    #         self.fig.canvas.flush_events()
    #         frame_count += 1