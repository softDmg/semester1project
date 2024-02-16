# import os
import sys
# import time
import wave

import cv2
import soundfile as sf
import librosa
import librosa.display
# import math
import numpy as np
import pandas as pd
import pyaudio
# import cv2
# import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QVBoxLayout
from PyQt5.QtWidgets import QRubberBand
# from pydub import AudioSegment
# from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from numpy.random import normal, uniform
from pydub import AudioSegment
from pydub.playback import play
from scipy.fft import fftfreq
from scipy.signal import butter, lfilter, stft, hilbert

from project import Ui_MainWindow


class AudioPlayerThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, audio_file_path):
        super(AudioPlayerThread, self).__init__()
        self.ui = None
        self.audio_file_path = audio_file_path


    def run(self):
        try:
            audio = AudioSegment.from_wav(self.audio_file_path)
            play(audio)
        except Exception as e:
            self.ui.statusLabel2.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error playing audio: {e}</b></p></html>')
        self.finished_signal.emit()


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)

        self.ax = None
        self.noisy_signal = None
        self.is_audio_playing = False
        self.is_noisy_audio_playing = False
        self.audio_player_thread = None
        self.noisy_audio_player_thread = None
        self.image_file_path = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.statusLabel.setText('<html><p style="font-size:25px; color:blue; background-color:white; text-align:center;"><b>Welcome</b></p></html>')
        self.channels = 2
        self.sample_rate = 44100
        self.image_file_path = None
        self.originalImage = None

        #Matplotlib figure
        self.figure = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self.ui.graphicsView3)

        # Set layout for graphicsView3
        layout = QVBoxLayout(self.ui.graphicsView3)
        layout.addWidget(self.canvas)


        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.ui.rightDisplayLabel)
        self.start_point = None
        self.original_image_backup = None

        #Deactivate objects on Image Processing Widget when app runs
        self.ui.categoryCombobox.hide()
        self.ui.grayscaleButton.hide()
        self.ui.basicGroupbox.hide()
        self.ui.advancedGroupbox.hide()
        self.ui.statusLabel2.clear()

        #Deactivate objects on Audio Processing Widget when app runs
        # self.ui.plotFreqButton.hide()
        self.ui.audioPlayButtonFrame.hide()
        self.ui.plotGroupBox.hide()
        self.ui.waveform.hide()
        self.ui.filter_groupbox.hide()
        self.ui.denoise_button.hide()
        self.ui.add_noise_groupBox.hide()

        # self.ui.spectral_factor_lineedit.hide()
        # self.ui.spectral_floor_lineedit.hide()
        # self.ui.spec_factor_label.hide()
        # self.ui.spec_floor_label.hide()
        #
        # self.ui.cutoff_freq_label.hide()
        # self.ui.cutoff_frequency_lineedit.hide()
        # self.ui.filter_order_label.hide()
        # self.ui.filter_order_lineedit.hide()

        #Connect slots
        self.ui.uploadAudioPushButton.clicked.connect(self.upload_audio)
        self.ui.recordAudioPushButton.clicked.connect(self.start_recording)
        self.ui.stopRecordPushButton.clicked.connect(self.stop_recording)
        self.ui.uploadImagePushButton.clicked.connect(self.upload_image)
        self.ui.categoryCombobox.currentIndexChanged.connect(self.operation)
        # self.ui.invertOnRadioButton.toggled.connect(self.update_inversion)
        self.ui.grayscaleButton.clicked.connect(self.grayScaling)
        self.ui.resetButton.clicked.connect(self.grayScaling)
        # self.ui.resetButton.clicked.connect(self.reset_image)
        # self.ui.playOriginalAudio.clicked.connect(self.play_original_audio)
        # self.ui.stopAudioPlayPushButton.clicked.connect(self.stop_audio_playback)
        self.ui.playOriginalAudio.clicked.connect(self.toggle_audio_playback)
        # self.ui.addNoiseButton.clicked.connect(self.addNoise)
        # self.ui.plotFreqButton.clicked.connect(self.plot_frequency_domain)
        # self.denoising_filter = self.butter_lowpass_filter
        self.ui.plotButton.clicked.connect(self.plot_audio)
        self.ui.signalEnergyButton.clicked.connect(self.plot_signal_energy)
        self.ui.blurSlider.valueChanged.connect(self.handle_blur_slider_change)
        self.ui.brightSlider.valueChanged.connect(self.handle_brightness_change)
        self.ui.addNoiseButton.clicked.connect(self.apply_noise)
        self.ui.playNoisyAudio.clicked.connect(self.toggle_noisy_audio_playback)
        self.ui.sharpenSlider.valueChanged.connect(self.handle_sharpness_change)
        self.ui.filterButton.clicked.connect(self.apply_median_filter)
        self.ui.cropImageButton.clicked.connect(self.crop_image)
        self.ui.rightDisplayLabel.installEventFilter(self)
        self.ui.undoCropButton.clicked.connect(self.undo_crop)
        self.ui.edgeDetectButton.clicked.connect(self.canny_edge_detection)
        self.ui.flipImageButton.clicked.connect(self.flip_image)
        self.ui.saveImageButton.clicked.connect(self.save_image)
        self.ui.redSlider.valueChanged.connect(self.handle_red_channel_change)
        self.ui.greenSlider.valueChanged.connect(self.handle_green_channel_change)
        self.ui.blueSlider.valueChanged.connect(self.handle_blue_channel_change)
        self.ui.loadPushButton.clicked.connect(self.dataLoad)
        self.ui.waveform.clicked.connect(self.reset_to_original_waveform)
        self.ui.denoise_button.clicked.connect(self.apply_audio_filter)
        self.ui.playNoisyAudio.clicked.connect(self.play_noisy_audio)
        self.ui.playDenoisedAudio.clicked.connect(self.play_denoise_audio)


        # For Low-Pass Filter parameters
        self.ui.cutoff_frequency_lineedit.setPlaceholderText("cutoff freq (e.g., 1000)")
        self.ui.filter_order_lineedit.setPlaceholderText("filter order (e.g., 5)")

        # For Spectral Subtraction parameters
        self.ui.spectral_floor_lineedit.setPlaceholderText("spectral floor (e.g., 0.001)")
        self.ui.spectral_factor_lineedit.setPlaceholderText("spectral factor (e.g., 0.1)")

        # You can also set default values if applicable
        # self.ui.cutoff_frequency_lineedit.setText("1000")
        # self.ui.filter_order_lineedit.setText("5")
        # self.ui.spectral_floor_lineedit.setText("0.001")
        # self.ui.spectral_factor_lineedit.setText("0.1")



    def start_recording(self):
        self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Recording....</b></p></html>')
        self.stop_audio_playback()
        self.duration = 15
        self.sample_rate = 44100
        self.channels = 2

        self.p = pyaudio.PyAudio()
        self.audio_format = pyaudio.paInt16


        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=1024)

        self.frames = []
        for i in range(0, int(self.sample_rate / 1024 * self.duration)):
            data = self.stream.read(1024)
            self.frames.append(data)

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()


        #Save the recorded audio
        file_path = "audio1.wav"
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Audio Recorded Successfully!</b></p></html>')

        #Update the audio file path
        self.audio_file_path = file_path

        #Plot the audio waveform
        self.plot_audio_waveform(file_path)

        # self.ui.plotFreqButton.show()
        self.ui.audioPlayButtonFrame.show()


    def upload_audio(self):
        self.ui.graphicsView.setScene(None)
        self.ui.graphicsView.clearFocus()
        self.ui.plotComboBox.setCurrentIndex(0)
        self.stop_audio_playback()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setOptions(options)
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3)")

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]

                if not hasattr(self, 'p') or self.p is None:
                    self.p = pyaudio.PyAudio()
                try:
                    uploaded_file_path = "uploaded_audio.wav"
                    wf = wave.open(uploaded_file_path, 'wb')
                    wf.setnchannels(self.channels)

                    with wave.open(file_path, 'rb') as audio_file:
                        wf.setsampwidth(audio_file.getsampwidth())
                        wf.setframerate(audio_file.getframerate())
                        wf.setnframes(audio_file.getnframes())
                        wf.setcomptype(audio_file.getcomptype(), audio_file.getcompname())

                        # Read and write audio data
                        audio_data = audio_file.readframes(audio_file.getnframes())
                        wf.writeframes(audio_data)

                    wf.close()

                    self.audio_file_path = uploaded_file_path

                    # self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Audio Uploaded Successfully!</b></p></html>')

                    signal, original_sampling_rate = librosa.load(self.audio_file_path, sr=None)
                    #Plot the audio waveform
                    self.plot_audio_waveform(self.audio_file_path)

                    self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Audio Upload Succesfully! - Signal Length: {len(signal)}</b></p></html>')


                except wave.Error as e:
                    self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error: {e}</b></p></html>')
                except Exception as e:
                    self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error: {e}</b></p></html>')

        self.ui.plotGroupBox.show()
        # self.ui.plotFreqButton.show()
        self.ui.audioPlayButtonFrame.show()


    def plot_audio_waveform(self, file_path):
        self.ui.graphicsView.setScene(None)
        self.ui.graphicsView.clearFocus()
        self.ui.waveform.show()
        self.ui.add_noise_groupBox.show()

        try:
            signal, sample_rate = librosa.load(file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=signal, sr=sample_rate)
            time_array = np.linspace(0, duration, num=len(signal))

            # Create a figure and axis
            fig, ax = plt.subplots()
            ax.plot(time_array, signal, color='b')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')

            # Display details in the top-left corner
            audio_info = f"Audio File: {file_path}\nSample Rate: {sample_rate} Hz\nDuration: {duration:.2f} seconds"
            ax.text(0.02, 0.98, audio_info, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

            # Set x-axis limits to match the duration of the audio file
            ax.set_xlim(0, duration)

            # Embed the plot in the graphicsView
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(900, 700)
            scene = QtWidgets.QGraphicsScene(self)
            scene.addWidget(canvas)
            self.ui.graphicsView.setScene(scene)

            return time_array, signal

        except Exception as e:
            print(f"Error loading audio file: {e}")
            # Handle the exception as needed

    def reset_to_original_waveform(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                # Load the audio file using librosa
                signal, _ = librosa.load(self.audio_file_path, sr=None)

                # Plot the original waveform
                self.plot_audio_waveform(signal)

            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error resetting to original waveform: {e}</b></p></html>')
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Reset to Original Waveform</b></p></html>')

    def play_original_audio(self):
        self.stop_audio_playback()
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Playing Original Audio</b></p></html>')

            #Create an instance of the audio player thread
            self.audio_player_thread = AudioPlayerThread(self.audio_file_path)
            self.audio_player_thread.finished_signal.connect(self.audio_playback_finished)

            #Starting thread
            self.audio_player_thread.start()
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Play</b></p></html>')

    def audio_playback_finished(self):
        self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Audio Playback Finished</b></p></html>')

    def stop_audio_playback(self):
        if self.audio_player_thread and self.audio_player_thread.isRunning():
            self.audio_player_thread.terminate()
            self.audio_player_thread.wait()
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Audio Playback Stopped</b></p></html>')

    def toggle_audio_playback(self):
        if self.is_audio_playing:
            self.stop_audio_playback()
        else:
            self.play_original_audio()

        self.is_audio_playing = not self.is_audio_playing

    def plot_audio(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                audio = AudioSegment.from_wav(self.audio_file_path)
                audio_data = np.array(audio.get_array_of_samples())

                plot_type = self.ui.plotComboBox.currentText()
                if plot_type == "Frequency Domain":
                    self.plot_frequency_domain()
                elif plot_type == "Histogram":
                    self.plot_histogram()
                elif plot_type == "Spectogram":
                    self.plot_spectogram()
                elif plot_type == "Signal Envelope":
                    self.plot_signal_envelope()
                else:
                    self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Invalid plot type selected</b></p></html>')
            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error plotting audio: {e}</b></p></html>')

    @staticmethod
    def auto_detect_nyquist(signal, original_sampling_rate):
        # Calculate the dominant frequency component
        frequencies = fftfreq(len(signal), 1/original_sampling_rate)
        magnitudes = np.abs(np.fft.fft(signal))
        dominant_frequency = frequencies[np.argmax(magnitudes)]

        #Convert frequencies to Hertz
        frequencies_in_hertz = frequencies * original_sampling_rate

        # Use the dominant frequency as the auto-detected Nyquist frequency
        auto_detected_nyquist = 2 * dominant_frequency

        return auto_detected_nyquist, frequencies_in_hertz, magnitudes


    def plot_frequency_domain(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                signal, original_sampling_rate = librosa.load(self.audio_file_path, sr=None)

                self.ui.statusLabel.setText("Plot Frequency Domain - Signal Length: " + str(len(signal)))

                # Auto-detect Nyquist frequency based on the loaded signal
                auto_detected_nyquist, frequencies, magnitudes = self.auto_detect_nyquist(signal, original_sampling_rate)

                positives_freqs = frequencies[frequencies >= 0]
                positive_magnitudes = magnitudes[:len(positives_freqs)]

                # Plot the frequency domain
                positive_frequencies = frequencies[frequencies >= 0]
                positive_magnitude_spectrum = positive_magnitudes[:len(positive_frequencies)]

                fig, ax = plt.subplots()
                # ax.plot(positive_frequencies, 10 * np.log10(np.abs(positive_magnitude_spectrum)), color='b')
                # ax.plot(positive_frequencies, 20 * np.log10(np.abs(positive_magnitude_spectrum)), color='b')
                ax.plot(frequencies, magnitudes, color='b')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
                # ax.set_ylim(0, 100)
                plt.title('Frequency Domain')
                plt.grid(True)

                # Display details in the top-left corner
                audio_info = f"Audio File: {self.audio_file_path}\nSample Rate: {original_sampling_rate} Hz\nDuration: {len(signal) / original_sampling_rate:.2f} seconds"
                ax.text(0.02, 0.98, audio_info, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

                if hasattr(self, 'toolbar'):
                    self.removeToolBar(self.toolbar)

                # Embed the plot in the graphicsView
                canvas = FigureCanvas(fig)
                canvas.setFixedSize(900, 700)
                scene = QtWidgets.QGraphicsScene(self)
                scene.addWidget(canvas)
                self.ui.graphicsView.setScene(scene)

                # Create a new toolbar and add it to the window
                self.toolbar = NavigationToolbar(canvas, self)
                self.addToolBar(self.toolbar)

                self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Frequency Domain Plotted Successfully!</b></p></html>')

            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error plotting frequency domain: {e}</b></p></html>')

        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Plot Frequency Domain</b></p></html>')

    def plot_histogram(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                # Load the audio using librosa for consistent sample extraction
                audio, _ = librosa.load(self.audio_file_path, sr=None, mono=True)
                samples = (audio * (2**15)).astype(np.int16)  # Convert to 16-bit integer for compatibility

                # Plot the histogram
                samples_positive = samples[samples >= 0]
                fig, ax = plt.subplots()
                ax.hist(samples_positive, bins='auto', color='b', alpha=0.5, rwidth=0.85, density=True)
                ax.set_xlabel('Amplitude')
                ax.set_ylabel('Probability Density')
                ax.set_yscale('log')
                plt.title('Signal Histogram')

                # Display details in the top-left corner
                audio_info = f"Audio File: {self.audio_file_path}\nSample Count: {len(samples)}\nMax Amplitude: {np.max(np.abs(samples))}"
                ax.text(0.02, 0.98, audio_info, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

                # Clear the previous plot in the graphicsView
                self.ui.graphicsView.setScene(None)

                # Embed the plot in the graphicsView
                canvas = FigureCanvas(fig)
                canvas.setFixedSize(900, 700)
                scene = QtWidgets.QGraphicsScene(self)
                scene.addWidget(canvas)
                self.ui.graphicsView.setScene(scene)

                # Create a new toolbar and add it to the window
                if hasattr(self, 'toolbar'):
                    self.removeToolBar(self.toolbar)
                self.toolbar = NavigationToolbar(canvas, self)
                self.addToolBar(self.toolbar)

                self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Histogram Plotted Successfully!</b></p></html>')

            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error plotting histogram: {e}</b></p></html>')
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Plot Histogram</b></p></html>')



    def plot_spectogram(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                # Read the audio file
                audio = AudioSegment.from_wav(self.audio_file_path)
                samples = np.array(audio.get_array_of_samples())

                # Perform Short-Time Fourier Transform (STFT)
                f, t, Zxx = stft(samples, fs=self.sample_rate, nperseg=1024)

                # Plot the spectrogram
                fig, ax = plt.subplots()
                c = ax.pcolormesh(t, f, 10 * np.log10(np.abs(Zxx) + 1e-10), shading='auto', cmap='viridis')
                fig.colorbar(c, ax=ax, label='Power/Frequency (dB/Hz)')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                plt.title('Signal Spectogram')

                # Check if the toolbar exists and remove it
                if hasattr(self, 'toolbar'):
                    self.removeToolBar(self.toolbar)

                # Embed the plot in the graphicsView
                canvas = FigureCanvas(fig)
                canvas.setFixedSize(900, 700)
                scene = QtWidgets.QGraphicsScene(self)
                scene.addWidget(canvas)
                self.ui.graphicsView.setScene(scene)

                # Create a new toolbar and add it to the window
                self.toolbar = NavigationToolbar(canvas, self)
                self.addToolBar(self.toolbar)

                self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Spectrogram Plotted Successfully!</b></p></html>')

            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error plotting spectrogram: {e}</b></p></html>')
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Plot Spectrogram</b></p></html>')

    def plot_signal_envelope(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                # Read the audio file
                audio = AudioSegment.from_wav(self.audio_file_path)
                samples = np.array(audio.get_array_of_samples())

                # Calculate the analytic signal using Hilbert transform
                analytic_signal = hilbert(samples)
                envelope = np.abs(analytic_signal)

                # Plot the signal envelope
                time_array = np.linspace(0, len(samples) / self.sample_rate, num=len(samples))
                fig, ax = plt.subplots()
                ax.plot(time_array, envelope, color='b')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Envelope')
                plt.title('Signal Envelop')

                if hasattr(self, 'toolbar'):
                    self.removeToolBar(self.toolbar)

                # Embed the plot in the graphicsView
                canvas = FigureCanvas(fig)
                canvas.setFixedSize(900, 700)
                scene = QtWidgets.QGraphicsScene(self)
                scene.addWidget(canvas)
                self.ui.graphicsView.setScene(scene)

                # Create a new toolbar and add it to the window
                self.toolbar = NavigationToolbar(canvas, self)
                self.addToolBar(self.toolbar)

                self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Signal Envelope Plotted Successfully!</b></p></html>')

            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error plotting signal envelope: {e}</b></p></html>')
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Plot Signal Envelope</b></p></html>')

    def plot_signal_energy(self):
        if hasattr(self, 'audio_file_path') and self.audio_file_path:
            try:
                # Read the audio file
                audio = AudioSegment.from_wav(self.audio_file_path)
                samples = np.array(audio.get_array_of_samples())

                # Calculate the signal energy
                energy = np.sum(np.square(samples))

                # Display the signal energy
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Signal Energy: {energy}</b></p></html>')

            except Exception as e:
                self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error calculating signal energy: {e}</b></p></html>')
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Audio File Available to Calculate Signal Energy</b></p></html>')


    def apply_noise(self):
        selected_noise = self.ui.noiseCombobox.currentText()

        # Load the audio signal
        signal, _ = librosa.load(self.audio_file_path, sr=None)

        noisy_signal = None
        if selected_noise == "Gaussian":
            noisy_signal = self.gaussian_noise(signal)
        elif selected_noise == "Uniform":
            noisy_signal = self.uniform_noise(signal)

        # Plot the audio waveform with noise
        if len(noisy_signal) > 0:
            noisy_path = "noisy.wav"
            sf.write(noisy_path, noisy_signal, _)
            print(f"Librosa version: {librosa.__version__}")
            if selected_noise == "Gaussian":
                self.plot_audio_waveform_with_noise(signal, noisy_signal, noise_type="Gaussian Noise")
            elif selected_noise == "Uniform":
                self.plot_audio_waveform_with_noise(signal, noisy_signal, noise_type="Uniform Noise")
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Noisy Signal is empty</b></p></html>')
        return noisy_signal, selected_noise

    def play_noisy_audio(self):
        try:
            _, _, noisy_path = self.apply_noise()
            self.ui.statusLabel.setText(f"Playing noisy audio from {noisy_path}")
            noisy_audio = AudioSegment.from_wav(noisy_path)
            play(noisy_audio)
        except Exception as e:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error playing noisy signal: {e}</b></p></html>')

    def plot_audio_waveform_with_noise(self, original_signal, noisy_signal, noise_type=""):
        self.ui.denoise_button.show()
        self.ui.filter_groupbox.show()
        # Create a Matplotlib figure and axes
        fig, ax = plt.subplots(figsize=(9, 5))

        # Plot the original signal
        librosa.display.waveshow(original_signal, alpha=0.5, label='Original Signal', color='blue')

        # Plot the noisy signal
        librosa.display.waveshow(noisy_signal, alpha=0.5, label='Noisy Signal', color='red')

        title = f"Waveform with {noise_type}"
        plt.title(title)

        # Create a Matplotlib canvas
        canvas = FigureCanvasQTAgg(fig)

        # Create a QWidget for embedding the Matplotlib canvas
        widget = QtWidgets.QWidget(self.ui.graphicsView)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(canvas)

        # Create a QGraphicsScene and set the QWidget as its view
        scene = QtWidgets.QGraphicsScene(self)
        scene.addWidget(widget)

        # Set the QGraphicsScene as the scene for the QGraphicsView
        self.ui.graphicsView.setScene(scene)

        # Display the Matplotlib canvas
        canvas.draw()

        # Show the QWidget containing the Matplotlib canvas
        widget.show()

    def gaussian_noise(self, signal, mean=0, std=0.1):
        noise = normal(mean, std, len(signal))
        return signal + noise

    def uniform_noise(self, signal, low=-0.1, high=0.1):
        noise = uniform(low, high, len(signal))
        return signal + noise

    def play_noisy_audio(self):
        self.stop_audio_playback()

        if hasattr(self, 'noisy_signal') and self.noisy_signal is not None:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Playing Noisy Audio</b></p></html>')

            # Create an instance of the audio player thread for noisy audio
            self.noisy_audio_player_thread = AudioPlayerThread(self.noisy_signal)
            self.noisy_audio_player_thread.finished_signal.connect(self.audio_playback_finished)

            # Starting thread for noisy audio
            self.noisy_audio_player_thread.start()
        else:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>No Noisy Audio Available to Play</b></p></html>')

    def toggle_noisy_audio_playback(self):
        if self.is_noisy_audio_playing:
            self.stop_noisy_audio_playback()
        else:
            self.play_noisy_audio()

        self.is_noisy_audio_playing = not self.is_noisy_audio_playing

    def stop_noisy_audio_playback(self):
        if self.noisy_audio_player_thread and self.noisy_audio_player_thread.isRunning():
            self.noisy_audio_player_thread.terminate()
            self.noisy_audio_player_thread.wait()
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:blue; background-color:white; text-align:center;"><b>Noisy Audio Playback Stopped</b></p></html>')

    def low_pass_filter(self, signal, sample_rate, cutoff_frequency, filter_order):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal
    def spectral_subtraction(self, signal, sample_rate, spectral_floor, spectral_factor):
            # Short-time Fourier Transform (STFT) to obtain the magnitude and phase
            stft_matrix = librosa.stft(signal)
            magnitude = np.abs(stft_matrix)

            # Estimate the noise magnitude spectrum (assuming it's stationary)
            noise_magnitude = np.median(magnitude, axis=1)

            # Apply spectral subtraction
            enhanced_magnitude = np.maximum(magnitude - spectral_factor * noise_magnitude[:, np.newaxis], spectral_floor)

            # Inverse STFT to obtain the enhanced signal
            enhanced_signal = librosa.istft(enhanced_magnitude * np.exp(1j * np.angle(stft_matrix)))

            return enhanced_signal

    def apply_audio_filter(self):
        filter_type = self.ui.filter_type_comboBox.currentText()

        try:
            # Get the original signal (assuming it's already loaded)
            signal, sample_rate = librosa.load(self.audio_file_path, sr=None)

            if filter_type == 'Low-Pass Filter':
                self.ui.cutoff_freq_label.show()
                self.ui.cutoff_frequency_lineedit.show()
                self.ui.filter_order_label.show()
                self.ui.filter_order_lineedit.show()


                # Wait for user input and retrieve values
                cutoff_frequency = float(self.ui.cutoff_frequency_lineedit.text())
                filter_order = float(self.ui.filter_order_lineedit.text())

                filtered_signal = self.low_pass_filter(signal, sample_rate, cutoff_frequency, filter_order)
                self.plot_audio_waveform_with_noise(signal, filtered_signal, "Low Pass Filter")

            elif filter_type == 'Spectral Subtraction':
                self.ui.spectral_factor_lineedit.show()
                self.ui.spectral_floor_lineedit.show()
                self.ui.spec_factor_label.show()
                self.ui.spec_floor_label.show()

                # Wait for user input and retrieve values
                spectral_floor = float(self.ui.spectral_floor_lineedit.text())
                spectral_factor = float(self.ui.spectral_factor_lineedit.text())

                # Apply spectral subtraction filter
                enhanced_signal = self.spectral_subtraction(signal, sample_rate, spectral_floor, spectral_factor)


                #Saving the denoise signal to a file
                denoise_path = "denoise.wav"
                sf.write(denoise_path, enhanced_signal, sample_rate)
                # librosa.output.write_wav(denoise_path, enhanced_signal, sample_rate)

                # Plot the original and enhanced signals
                self.plot_audio_waveform_with_noise(signal, enhanced_signal, "Spectral Subtraction Filtered")

                return denoise_path
        except ValueError as ve:
            # Handle the ValueError (e.g., invalid user input) here
            self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error: {ve}</b></p></html>')

        except Exception as e:
            # Handle other exceptions here
            self.ui.statusLabel.setText(f'<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error applying filter: {e}</b></p></html>')

    def play_denoise_audio(self):
        try:
            _, _, denoise_path = self.apply_audio_filter()
            self.ui.statusLabel.setText(f"Playing denoised audio from{denoise_path}")
            denoise_audio = AudioSegment.from_wav(denoise_path)
            play(denoise_audio)
        except Exception as e:
            self.ui.statusLabel.setText('<html><p style="font-size:15px; color:red; background-color:white; text-align:center;"><b>Error playing denoised signal</b></p></html>')




############################################################################################################################
################################################ IMAGE PROCESSING ##########################################################
############################################################################################################################

    def upload_image(self):
        # self.ui.invertOnRadioButton.setChecked(False)
        # self.ui.invertOffRadioButton.setChecked(False)
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        self.ui.categoryCombobox.show()
        self.ui.grayscaleButton.show()
        # self.ui.thresholdButton.show()

        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setNameFilter("Image Files (*.png *.jpg *.bmp)")
        image_file_paths, _ = file_dialog.getOpenFileNames()

        if image_file_paths:
            self.image_file_path = image_file_paths[0]
            self.originalImage = cv2.imread(self.image_file_path)
            self.ui.statusLabel2.setText(f'<html><p style="font-size:15px; color:green; background-color:white; text-align:center;"><b>Image Uploaded Successfully!</b></p></html>')
            # self.ui.playbackFrame.show()

            # Display image on a display widget
            pixmap = QtGui.QPixmap(self.image_file_path)
            pixmap = pixmap.scaled(self.ui.displayLabel.size(), QtCore.Qt.KeepAspectRatio)

            self.ui.displayLabel.setPixmap(pixmap)
            self.ui.displayLabel.setScaledContents(True)
        else:
            self.ui.statusLabel2.setText(
                "<html><font color='red' size='6' face='Arial' style=background-color:white;><b>No Image Selected!</b></font></html>")


    def grayScaling(self):
        # self.ui.invertOnRadioButton.setChecked(False)
        # self.ui.invertOffRadioButton.setChecked(False)
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        self.ui.rightDisplayLabel.clear()
        if self.image_file_path is not None:
            img = cv2.imread(self.image_file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(image)
                self.ui.rightDisplayLabel.setPixmap(pixmap)
                self.ui.rightDisplayLabel.setScaledContents(True)
                self.ui.categoryCombobox.show()
            else:
                self.ui.statusLabel2.setText(
                    "<html><font color='red' size='6' face='Arial' style=background-color:white;><b>Image Read Error!!</b></font></html>")
        else:
            self.ui.statusLabel2.setText(
                "<html><font color='red' size='6' face='Arial' style=background-color:white;><b>Image File Path is None!</b></font></html>")

    def reset_image(self):
        self.ui.brightSlider.setValue(self.ui.brightSlider.minimum())
        # self.ui.invertOnRadioButton.setChecked(False)
        # self.ui.invertOffRadioButton.setChecked(False)
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        if self.originalImage is not None:
            height, width, channel = self.originalImage.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.originalImage.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.rightDisplayLabel.setScaledContents(True)
            self.ui.statusLabel2.setText("<html><font color='blue' size='3' face='Arial'><b>Image Reset Successfully!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("No original image available to reset")

    def handle_category(self):
        selected_category = self.ui.categoryCombobox.currentText()
        self.ui.colorChannelGroupbox.show()
        if selected_category == "basic":
            self.ui.basicGroupbox.show()
            self.ui.advancedGroupbox.hide()
        elif selected_category == "advanced":
            self.ui.advancedGroupbox.show()
            self.ui.basicGroupbox.hide()
        else:
            self.ui.statusLabel2.setText("No choice made")

    def handle_blur_slider_change(self, value):
        self.ui.brightSlider.setValue(self.ui.brightSlider.minimum())
        if self.originalImage is not None:
            # Ensure that the blur strength is a positive odd integer
            blur_strength = max(value, 1)
            blur_strength = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength

            # Apply the Gaussian blur operation to the image
            blurred_image = cv2.GaussianBlur(self.originalImage, (blur_strength, blur_strength), 0)

            # Check the color format of the original image
            if len(blurred_image.shape) == 3 and blurred_image.shape[2] == 3:
                # Convert BGR to RGB format
                blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

            # Convert OpenCV image to QImage
            height, width, channel = blurred_image.shape
            bytes_per_line = 3 * width  # Assuming 3 channels (RGB)
            q_image = QImage(blurred_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap and set it on the QLabel
            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)

            self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Image Blurred Successfully!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available to blur</b></font></html>")

    def handle_sharpness_change(self, value):
        self.ui.brightSlider.setValue(self.ui.brightSlider.minimum())
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        if self.originalImage is not None:
            sharpness_strength = max(value, 0)

            sharpened_image = self.sharpen_image(self.originalImage, sharpness_strength)

            if len(sharpened_image.shape) == 3 and sharpened_image.shape[2] == 3:

                sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

                height, width, channel = sharpened_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(sharpened_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)
                self.ui.rightDisplayLabel.setPixmap(pixmap)

                self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Image Sharpened Successfully!</b></font></html>")

            else:
                self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available to sharpen</b></font></html>")

    def sharpen_image(self, image, strength):

        kernel = np.array([[-1,-1,-1],
                           [-1,9,-1],
                           [-1,-1,-1]])

        kernel = kernel * strength

        sharpen_image = cv2.filter2D(image, -1, kernel)

        sharpen_image = np.clip(sharpen_image, 0, 255)

        return sharpen_image


    def handle_brightness_change(self, brightness_value):
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        if self.originalImage is not None:
            # Adjust the brightness of the original image
            adjusted_image = self.adjust_brightness(self.originalImage, brightness_value)

            if len(adjusted_image.shape) == 3 and adjusted_image.shape[2] == 3:
                adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

            # Display the adjusted image
            height, width, channel = adjusted_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.statusLabel2.setText(f"<html><font color='white' size='3' face='Arial'><b>Brightness Adjusted: {brightness_value}</b></font></html>")
        else:
            self.ui.statusLabel2.setText("No original image available to adjust brightness")

    def adjust_brightness(self, image, brightness_value):
        # Adjust the brightness using OpenCV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, brightness_value)
        v = np.clip(v, 0, 255)
        hsv = cv2.merge((h, s, v))
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return adjusted_image


    def apply_filter(self):
        selected_filter = self.filterCombobox2.currentText()

        if selected_filter == "Median Filter":
            self.apply_median_filter()
        elif selected_filter == "Vertical Filter":
            self.apply_vertical_filter()
        elif selected_filter == "Average Filter":
            self.apply_average_filter()
        elif selected_filter == "Laplacian Filter":
            self.apply_laplacian_filter()

    def apply_median_filter(self):
        if self.originalImage is not None:
            gray_image = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)


            kernel_size = 3
            median_filtered_image = cv2.medianBlur(gray_image, kernel_size)

            bytes_per_line = median_filtered_image.shape[1] * median_filtered_image.shape[2] if len(median_filtered_image.shape) == 3 else median_filtered_image.shape[1]
            q_image = QImage(median_filtered_image.data, median_filtered_image.shape[1], median_filtered_image.shape[0], bytes_per_line, QImage.Format_Grayscale8 if len(median_filtered_image.shape) == 2 else QImage.Format_RGB888)


            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.rightDisplayLabel.setScaledContents(True)

            self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Median Filter Applied Successfully!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available to apply median filter</b></font></html>")


    def apply_vertical_filter(self):
        if self.originalImage is not None:
            gray_image = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)

            #Apply Sobel Filter for vertical edge detection
            sobel_filtered_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            abs_sobel_filtered_image = cv2.convertScaleAbs(sobel_filtered_image)

            if len(abs_sobel_filtered_image.shape) == 2:
                q_image = QImage(abs_sobel_filtered_image.data, abs_sobel_filtered_image.shape[1], abs_sobel_filtered_image.shape[0],  abs_sobel_filtered_image.shape[1], QImage.Format_Grayscale8)
            else:
                q_image = QImage(abs_sobel_filtered_image.data, abs_sobel_filtered_image.shape[1], abs_sobel_filtered_image.shape[0], 3 * abs_sobel_filtered_image.shape[1], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.rightDisplayLabel.setScaledContents(True)

            self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Vertical Filter Applied Successfully!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available to apply vertical filter</b></font></html>")


    def apply_average_filter(self):
        pass
    def apply_laplacian_filter(self):
        pass


    def crop_image(self):
        if self.originalImage is not None:
            # Display the original image for user interaction
            pixmap = QtGui.QPixmap.fromImage(self.array_to_qimage(self.originalImage))
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.rightDisplayLabel.setScaledContents(True)

            # Set up an event filter to capture mouse events on the image label
            self.ui.rightDisplayLabel.installEventFilter(self)

            self.ui.statusLabel2.setText("<html><font color='blue' size='3' face='Arial'><b>Select the region to crop...</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available to crop</b></font></html>")

    def eventFilter(self, obj, event):
        if obj == self.ui.rightDisplayLabel and event.type() == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.LeftButton:
                self.start_point = event.pos()
                self.rubber_band.setGeometry(QtCore.QRect(self.start_point, QtCore.QSize()))
                self.rubber_band.show()
                return True
        elif obj == self.ui.rightDisplayLabel and event.type() == QtCore.QEvent.MouseMove:
            if self.start_point is not None:
                self.rubber_band.setGeometry(QtCore.QRect(self.start_point, event.pos()).normalized())
                return True
        elif obj == self.ui.rightDisplayLabel and event.type() == QtCore.QEvent.MouseButtonRelease:
            if event.button() == QtCore.Qt.LeftButton and self.start_point is not None:
                # Get the end point
                end_point = event.pos()

                # Map the points from the label to the parent widget
                start_point_img = self.ui.rightDisplayLabel.mapToParent(self.start_point)
                end_point_img = self.ui.rightDisplayLabel.mapToParent(end_point)

                # Convert points to image coordinates
                start_point_img = self.map_to_image_coordinates(start_point_img)
                end_point_img = self.map_to_image_coordinates(end_point_img)

                # Ensure that the rectangle is valid
                start_x, start_y = start_point_img.x(), start_point_img.y()
                end_x, end_y = end_point_img.x(), end_point_img.y()

                x = min(start_x, end_x)
                y = min(start_y, end_y)
                width = abs(end_x - start_x)
                height = abs(end_y - start_y)

                # Crop the image using the selected rectangle
                cropped_image = self.originalImage[y:y + height, x:x + width]

                # Display the cropped image
                pixmap = QtGui.QPixmap.fromImage(self.array_to_qimage(cropped_image))
                self.ui.rightDisplayLabel.setPixmap(pixmap)
                self.ui.rightDisplayLabel.setScaledContents(True)

                self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Image Cropped Successfully!</b></font></html>")
                self.start_point = None
                return True

        return super().eventFilter(obj, event)

    def map_to_image_coordinates(self, point):
        if self.originalImage is not None:
            label_rect = self.ui.rightDisplayLabel.rect()
            image_width, image_height, _ = self.originalImage.shape

            # Calculate the scaling factors
            scale_x = image_width / label_rect.width()
            scale_y = image_height / label_rect.height()

            # Map the point from label coordinates to image coordinates
            x = int((point.x() - label_rect.left()) * scale_x)
            y = int((point.y() - label_rect.top()) * scale_y)

            return QtCore.QPoint(x, y)

        return QtCore.QPoint(0, 0)

    def undo_crop(self):
        if self.original_image_backup is not None:
            # Restore the original image
            self.originalImage = self.original_image_backup.copy()

            # Display the original image
            pixmap = QtGui.QPixmap.fromImage(self.array_to_qimage(self.originalImage))
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.rightDisplayLabel.setScaledContents(True)

            self.ui.statusLabel2.setText("<html><font color='blue' size='3' face='Arial'><b>Undo Crop Successful!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No crop to undo</b></font></html>")



    def crop_opencv(self, image, crop_rect):
        x, y, w, h = crop_rect
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image
    def array_to_qimage(self, image_array):
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        image_data = image_array.tobytes()
        q_image = QtGui.QImage(image_data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        return q_image

    def canny_edge_detection(self):
        if self.originalImage is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(self.originalImage, cv2.COLOR_BGR2GRAY)

            # Perform Canny edge detection
            edges = cv2.Canny(gray_image, 50, 150)

            # Convert edges to RGB format
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Display the edges image
            height, width, channel = edges_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(edges_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)
            self.ui.rightDisplayLabel.setScaledContents(True)

            self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Canny Edge Detection Applied Successfully!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available for edge detection</b></font></html>")

    def flip_image(self):
            if self.originalImage is not None:
                # Flip the image horizontally
                flipped_image = cv2.flip(self.originalImage, 1)

                # Display the flipped image
                height, width, channel = flipped_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(flipped_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.ui.rightDisplayLabel.setPixmap(pixmap)
                self.ui.rightDisplayLabel.setScaledContents(True)

                self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Image Flipped Successfully!</b></font></html>")
            else:
                self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No original image available to flip</b></font></html>")

    def save_image(self):
        if self.ui.rightDisplayLabel.pixmap() is not None:
            save_dialog = QFileDialog()
            save_dialog.setNameFilter("PNG Image (*.png);;JPEG Image (*.jpg);;Bitmap Image (*.bmp)")
            save_dialog.setDefaultSuffix('png')  # Default file extension

            save_path, _ = save_dialog.getSaveFileName(self.ui.centralwidget, 'Save Image', '', "Images (*.png *.jpg *.bmp)")

            if save_path:
                # Retrieve the displayed image from the label
                displayed_pixmap = self.ui.rightDisplayLabel.pixmap()
                displayed_image = displayed_pixmap.toImage()

                # Save the image to the specified file path
                displayed_image.save(save_path)
                self.ui.statusLabel2.setText("<html><font color='green' size='3' face='Arial'><b>Image Saved Successfully!</b></font></html>")
        else:
            self.ui.statusLabel2.setText("<html><font color='red' size='3' face='Arial'><b>No image to save</b></font></html>")

    def handle_red_channel_change(self, value):
        self.ui.brightSlider.setValue(self.ui.brightSlider.minimum())  # Reset brightness slider
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())  # Reset blur slider
        self.ui.sharpenSlider.setValue(self.ui.sharpenSlider.minimum())  # Reset sharpness slider

        self.update_image(red=value)

    def handle_green_channel_change(self, value):
        self.ui.brightSlider.setValue(self.ui.brightSlider.minimum())
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        self.ui.sharpenSlider.setValue(self.ui.sharpenSlider.minimum())

        self.update_image(green=value)

    def handle_blue_channel_change(self, value):
        self.ui.brightSlider.setValue(self.ui.brightSlider.minimum())
        self.ui.blurSlider.setValue(self.ui.blurSlider.minimum())
        self.ui.sharpenSlider.setValue(self.ui.sharpenSlider.minimum())

        self.update_image(blue=value)

    def update_image(self, red=0, green=0, blue=0):
        if self.originalImage is not None:

            adjusted_image = self.adjust_color_channels(self.originalImage, red, green, blue)

            #Display the adjusted image
            height, width, channel = adjusted_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(adjusted_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.ui.rightDisplayLabel.setPixmap(pixmap)

    def adjust_color_channels(self, image, red=0, green=0, blue=0):
        # Adjust the individual color channels using OpenCV
        image[:, :, 0] = np.clip(image[:, :, 0] + blue, 0, 255)  # Blue channel
        image[:, :, 1] = np.clip(image[:, :, 1] + green, 0, 255)  # Green channel
        image[:, :, 2] = np.clip(image[:, :, 2] + red, 0, 255)    # Red channel
        return image




##############################################################################################################################
######################################################## NEUTRAL #############################################################
##############################################################################################################################



    def operation(self):
        selected_operation = self.ui.categoryCombobox.currentText()
        if selected_operation == "Basic":
            self.ui.basicGroupbox.show()
            self.ui.advancedGroupbox.hide()
            # self.ui.invertGroupbox.show()
        elif selected_operation == "Advanced":
            self.ui.basicGroupbox.hide()
            self.ui.advancedGroupbox.show()
        else:
            self.ui.statusLabel2.setText("No Selection")

##############################################################################################################################
######################################################## DATA PROCESSING #############################################################
##############################################################################################################################

    def handleDatasetType(self):
        selected_type = self.ui.dataTypeCombobox.currentText()
        if selected_type == "CSV":
            self.ui.meanButton.show()
            self.ui.medianButton.show()
        elif selected_type == "XLSX":
            pass
        elif selected_type == "JSON":
            pass
        elif selected_type == "SQL":
            pass
        elif selected_type == "TSV":
            pass

    def dataLoad(self):
        selected_type = self.ui.dataTypeCombobox.currentText()
        if selected_type == "CSV":
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(None, "Open CSV File", "", "CSV Files (*.csv)")
            if file_path:
                self.loadCSVData(file_path)

    def loadCSVData(self, file_path):
        print(f"Loadin CSV data from {file_path}")
        if file_path:
            df = pd.read_csv(file_path)

            # figure, self.ax = plt.subplots()
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(df.iloc[:, 0])  # Assuming the first column contains the ECG signal
            ax.set_title('ECG Signal')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')

            self.canvas.draw()
        else:
            print("File selection canceled")







if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
