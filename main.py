# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:15:17 2022

@author: bcary


TODO:        
-make video desc text wrap

-add indicator if dlc file already loaded

-move self.videoplayer.fps to just self.fps?

-consolidate code repetitions especially in model predict

-when you frame jump it doesnt update slider bounds

-add option in config to read fps from video

"""

DEBUG_ON = False

import os
import sys
import cv2
import csv
import yaml
import pickle
from datetime import datetime

from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication, QLabel, QVBoxLayout,
                             QHBoxLayout, QSlider, QGridLayout, QPushButton, QAction,
                             QFileDialog, QSizePolicy, QGraphicsLineItem, QGraphicsRectItem,
                             QGraphicsSimpleTextItem, QLineEdit, QInputDialog, QStyle,
                             QGraphicsTextItem, QSplitter)
from PyQt5.QtGui import QPixmap, QColor, QImage, QIcon, QPainter, QPen, QFont
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QDir, QSize

# from custom_widgets import 
from style import STYLESHEET, WIDGET_STYLES
import FeatureExtraction
import TrainModel_RF
import video_reader

from sklearn.decomposition import PCA

import random
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import use as pltuse
pltuse('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from typing import Union

# # for plotting in separate window
# from IPython import get_ipython

# if get_ipython():
#     get_ipython().run_line_magic('matplotlib', 'qt')



def open_config(config_path):
    # parses yaml file into self.config dictionary
    
    with open(config_path, 'r') as stream:
        try:
            yaml_data = yaml.safe_load(stream)
            
        except yaml.YAMLError as exc:
            print(exc)
            
    return yaml_data


def plot_full_labels(labels, c_map, xlims):
    # uses matplotlib to plot "ethogram" of entire video 
    
    labels = np.array(labels)
    num_labels = labels.shape[0]
    uk_labels = np.arange(num_labels)
    image = np.ones([max(uk_labels)+1,labels.shape[1],3])*255
    for ind, l in enumerate(uk_labels):
        i_inds = np.where(labels[l,:] == 1)[0]  
        image[l,i_inds,0:3] = c_map[l+1,:-1]*255
    
    image = image.astype(np.uint8)

    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(image, interpolation='nearest', aspect='auto')

    plt.xlim(xlims)
    ax.invert_yaxis()
    
    ax.set_ylabel('Labels')
    ax.set_xlabel('Frame number')
    plt.show()


def frame_to_time_text(frame_num, fps):
    # using frame number and fps create mins:secs:ms string notation
    
    total_seconds = frame_num/fps
    minutes = int(np.floor(total_seconds/60))
    seconds = int(total_seconds % 60)
    sec_dec = int((total_seconds % 1)*1000)
    
    text = f'{minutes:02d}:{seconds:02d}:{sec_dec:03d}'

    return text


# class MplCanvas(FigureCanvas):

#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = plt.Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super(MplCanvas, self).__init__(fig)







class VideoWindow(QtWidgets.QGraphicsView):
    def __init__(self, parent=None, video_path=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("video player") 
        self.parent = parent
        
        self.setStyleSheet("background:transparent;")
        
        # default params
        self.disply_width = 640
        self.display_height = 480
        self.fps = 30
        
        self.frame_i = 0
        self.left_range = 0
        self.right_range = 0
        self.frame_count = 0
        
        self._zoom = 0
        
        # create a text label
        self.textLabel = QLabel('Video player')
        # self.textLabel.setSizePolicy(QSizePolicy.Preferred,
        #         QSizePolicy.Maximum)
        
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(self.left_range, self.right_range)
        
        self.slider_backward_but = QPushButton()
        self.slider_backward_but.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_ArrowBack')))
        self.slider_backward_but.setIconSize(QSize(20,20))
        
        self.slider_forward_but = QPushButton()
        self.slider_forward_but.setIcon(self.style().standardIcon(getattr(QStyle, 'SP_ArrowForward')))
        self.slider_forward_but.setIconSize(QSize(20,20))

        self.slider_minus_but = QPushButton('')
        self.slider_minus_but.setIcon(QIcon(os.path.join('icons','minus.jpg')))
        self.slider_minus_but.setIconSize(QSize(20,20))
        
        self.slider_plus_but = QPushButton('')
        self.slider_plus_but.setIcon(QIcon(os.path.join('icons','plus.png')))
        self.slider_plus_but.setIconSize(QSize(20,20))        
        
        self.posTextBox = QLineEdit('0')
        self.posTextBox.resize(50, 25)
        self.posTextBox.setMaximumWidth(50)
        self.posTextBox.setFont(QFont('Arial', 12))
        
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        # self.videoView.setScene(self._scene)
        self.setScene(self._scene)

        # self.setMinimumSize(QSize(640, 480))
        
        if video_path:
            # get first frame
            self.loadPosition(0)
            
            # convert image to Qt format
            qt_img = self.convert_cv_qt(self.cv_img)
            # set the image image to the grey pixmap
            self.image_label.setPixmap(qt_img)            
        else:
            # create a grey pixmap
            grey = QPixmap(640, 480)
            grey.fill(QColor('darkGray'))
        
        
    def displayFrame(self, position):
        # print(f'position: {position}')
        # print(f'frame_i: {self.frame_i}')
        
        # can speed up run time by only setting video position when it's not the very next frame in queue
        # known issue in jumping while going back frames due to jumping to keyframes this is an issue with the videos compression
        
        
        # if position != self.frame_i + 1:
        #     # self.video.set(cv2.CAP_PROP_POS_FRAMES,position)
            
        #     pose_msec = (float(position) * 1000) / self.fps
        #     print(f'pose msec: {pose_msec}')
            
        #     self.video.set(cv2.CAP_PROP_POS_MSEC,pose_msec)
            
        # ret, self.cv_img = self.video.read()
        
        # curr_frame = self.video.get(cv2.CAP_PROP_POS_FRAMES)
        # print(f'video frame: {curr_frame}')
        
        # if position != self.frame_i + 1:
        #     self.cv_img = self.reader.request_frame(position)
                
        self.read_img = self.reader.request_frame(position)
        self.cv_img = self.read_img.image
        
        self.frame_i = position
    
        # if ret==True:
        # convert image to Qt format
        qt_img = self.convert_cv_qt(self.cv_img)
        
        self._photo.setPixmap(qt_img)    
        
        self.update()
                    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # from https://github.com/docPhil99/opencvQtdemo/blob/master/staticLabel2.py
        
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = int(ch * w)
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        
        return QPixmap.fromImage(p)

    def openFile(self, vid_path):
        self.video = cv2.VideoCapture(vid_path)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.frame_count = 1800
        
        self.disply_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        self.display_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # self.fps = self.video.get(cv2.CAP_PROP_FPS)
        
        self.left_range = 0
        self.right_range = self.frame_count
        self.positionSlider.setRange(self.left_range, self.right_range)
        
        # attempting to use video reader code
        self.reader = video_reader.VideoReader(path=vid_path)
        
        
        
        print("Video frame count: {}".format(self.frame_count))
        print("Video dimensions: {} width by {} height".format(self.disply_width, self.display_height))
        print("Video frames per second: {}".format(self.fps))
        
    def wheelEvent(self, event):
        # https://stackoverflow.com/questions/58965209/zoom-on-mouse-position-qgraphicsview
        if hasattr(self, 'video'):
            if event.angleDelta().y() > 0:
                factor = 1.1
                self._zoom += 1
            else:
                factor = 0.9
                self._zoom -= 1
            if self._zoom > 0:
                view_pos = event.pos()
                scene_pos = self.mapToScene(view_pos)
                self.centerOn(scene_pos)
                self.scale(factor, factor)
                delta = self.mapToScene(view_pos) - self.mapToScene(self.viewport().rect().center())
                self.centerOn(scene_pos - delta)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0
                self.fitInView()     
                
    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            # if self.hasPhoto():
            unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
            # print(factor, viewrect, scenerect)
            self.scale(factor, factor)
            self._zoom = 0
            
        
class LabelViewer(QtWidgets.QGraphicsView):
    
    def __init__(self, parent=None, labels=[], num_labels=0, c_map = [], fps=20):
        super().__init__()
        
        # handle class params
        self.parent = parent
        # labels = np.array(random.choices(range(-1,5),k=1800))
        if not labels.size:
            self.view_labels = -np.ones(500)
        else:
            self.view_labels = labels
        self.num_labels = num_labels # this number of labels does NOT include unlabeled and no behavior
        self.cmap = c_map
        self.fps = fps
        
        self.box_height = 60
        self.box_width = 7
        self.cursor_overshoot = -4
        self.line_width = 2
        self.window_len = 125 # needs to be even for cursor to perfectly line up
        
        if self.window_len > self.view_labels.shape[1]:
            self.window_len = int(np.round(len(self.view_labels)/2))
            
        if self.window_len % 2 != 0:
            self.window_len += 1

        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        self._scene = QtWidgets.QGraphicsScene(self)
        # init_bg = QPixmap(len(self.labels)*self.box_width, self.box_height)
        # init_bg.fill(Qt.white)
                
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)        
        self.setScene(self._scene)
        self.setMaximumHeight(self.box_height+26) # +26 to compensate some for the presence of added scrollbars

        self.frame_i = 0
        
        # self.image = self.array_to_image(self.labels)
        self.drawCursor()
        self.update_image()
        
    def drawCursor(self):
        pen = QPen(Qt.black, self.line_width)

        self.cursor = QGraphicsRectItem(self.frame_i*self.box_width, 0, self.box_width, self.box_height)        
        self.cursor.setPen(pen)
        self._scene.addItem(self.cursor)
        
        self.cursor_text = QGraphicsSimpleTextItem()
        cursor_font = QFont('Arial', 8)
        cursor_font.setBold(True)
        self.cursor_text.setFont(cursor_font)
        self.cursor_text.setPos(self.box_width*2, self.box_height/5)
        
        labels_i = np.where(self.view_labels[:,self.frame_i]==1)[0]
        cursor_time = frame_to_time_text(self.frame_i, self.fps)
        cursor_str = (f'Frame: {self.frame_i:.0f} \n'
                      f'{cursor_time} \n '
                      f'Label: {labels_i}')
        self.cursor_text.setText(cursor_str)
        self._scene.addItem(self.cursor_text)
        
    def update_image(self):        
        pos = self.frame_i
        last_wind_pos = self.view_labels.shape[1] - np.floor(self.window_len/2)
        
        if pos < np.floor(self.window_len/2):
            rel_pos = pos
            label_window = self.view_labels[:,:self.window_len]
        elif pos >= last_wind_pos:
            rel_pos = pos - last_wind_pos + np.floor(self.window_len/2)
                          
            label_window = self.view_labels[:,-self.window_len:]
        else:
            rel_pos = np.floor(self.window_len/2)
            label_window = self.view_labels[:,int(pos-np.floor(self.window_len/2)):
                                       int(pos+np.floor(self.window_len/2))] 
                
        self.cursor.setRect(rel_pos*self.box_width, 0, self.box_width, self.box_height)        
        self.cursor_text.setPos(rel_pos*self.box_width + self.box_width*2, self.box_height/5)
        
        labels_i = np.where(self.view_labels[:,self.frame_i]==1)[0]
        cursor_time = frame_to_time_text(self.frame_i, self.fps)
        cursor_str = (f'Frame: {self.frame_i:.0f} \n'
                      f'{cursor_time} \n '
                      f'Label: {labels_i}')
        self.cursor_text.setText(cursor_str)

        self.image = self.array_to_image(label_window)
        qpixmap = self.convert_cv_qt(self.image)
        
        self.qpixmap = qpixmap
        self._photo.setPixmap(self.qpixmap)
        self.fitInView(self._photo)
        self.update()   
    
    def array_to_image(self, array: np.ndarray, alpha: Union[float, int, np.ndarray] = None):        

        labels = np.array(array)
        num_labels = labels.shape[0]
        uk_labels = np.arange(num_labels)
        image = np.ones([max(uk_labels)+1,labels.shape[1],3])*255
        for ind, l in enumerate(uk_labels):
            i_inds = np.where(labels[l,:] == 1)[0]  
            image[l,i_inds,0:3] = self.cmap[l+1,:-1]*255
        
        image = image.astype(np.uint8)
        return image
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # from https://github.com/docPhil99/opencvQtdemo/blob/master/staticLabel2.py
        
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.box_width*cv_img.shape[1], self.box_height)
        
        return QPixmap.fromImage(p)    
    
        
class FullLabelViewer(QtWidgets.QGraphicsView):
    
    def __init__(self, parent=None, labels=[], num_labels=0, c_map = []):
        super().__init__()
        
        # handle class params
        self.parent = parent
        # labels = np.array(random.choices(range(-1,5),k=1800))
        if not labels.size:
            self.view_labels = -np.ones(500)
        else:
            self.view_labels = labels
        self.num_labels = num_labels # this number of labels does NOT include unlabeled and no behavior
        self.cmap = c_map
        
        self.box_height = 25
        self.box_width = 4
        self.cursor_overshoot = -4
        self.line_width = 2
        self.window_len = 600 # needs to be even for cursor to perfectly line up
        
        if self.window_len > self.view_labels.shape[1]:
            self.window_len = int(np.round(len(self.view_labels)/2))
            
        if self.window_len % 2 != 0:
            self.window_len += 1
            
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        # self.fitInView(self._photo)
        self.setMaximumHeight(self.box_height+5)

        self.frame_i = 0
        
        # self.image = self.array_to_image(self.labels)
        self.drawCursor()
        self.update_image()
        
    def drawCursor(self):
        pen = QPen(Qt.black, self.line_width)
        self.cursor = QGraphicsRectItem(self.frame_i, 0, self.box_width, self.box_height)        
        self.cursor.setPen(pen)
        self._scene.addItem(self.cursor)
        
    def update_image(self):        
        pos = self.frame_i
        last_wind_pos = self.view_labels.shape[1] - np.floor(self.window_len/2)
        
        if pos < np.floor(self.window_len/2):
            rel_pos = pos
            label_window = self.view_labels[:,:self.window_len]
        elif pos >= last_wind_pos:
            rel_pos = pos - last_wind_pos + np.floor(self.window_len/2)
                          
            label_window = self.view_labels[:,-self.window_len:]
        else:
            rel_pos = np.floor(self.window_len/2)
            label_window = self.view_labels[:,int(pos-np.floor(self.window_len/2)):
                                       int(pos+np.floor(self.window_len/2))] 
                
        self.cursor.setRect(rel_pos*self.box_width, 0, self.box_width, self.box_height)        
        
        self.image = self.array_to_image(label_window)
        qpixmap = self.convert_cv_qt(self.image)
        self.qpixmap = qpixmap
        self._photo.setPixmap(self.qpixmap)
        self.fitInView(self._photo)
        self.update()   
    
    def array_to_image(self, array: np.ndarray, alpha: Union[float, int, np.ndarray] = None):        
    
        labels = np.array(array)
        num_labels = labels.shape[0]
        uk_labels = np.arange(num_labels)
        image = np.ones([max(uk_labels)+1,labels.shape[1],3])*255
        for ind, l in enumerate(uk_labels):
            i_inds = np.where(labels[l,:] == 1)[0]  
            image[l,i_inds,0:3] = self.cmap[l+1,:-1]*255
        
        image = image.astype(np.uint8)
        return image
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        # from https://github.com/docPhil99/opencvQtdemo/blob/master/staticLabel2.py
        
        rgb_image = cv_img
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(self.frameGeometry().width(), self.box_height)
        p = convert_to_Qt_format.scaled(self.box_width*cv_img.shape[1], self.box_height)
        
        return QPixmap.fromImage(p)   
      
    
class LabelButtons(QVBoxLayout):
    
    def __init__(self, parent=None, label_names=[], cmap=[]):
        super().__init__()
        
        self.parent = parent
        self.label_names = label_names
        self.cmap = cmap
        
        self.short_lets = ['q','w','e','r','t','a','s','d','f','g','z','x','c','v']
        buffer_lets = ['']*100
        [self.short_lets.append(x) for x in buffer_lets]
        
        self.label = QLabel('Manual Behavioral Labels')
        label_font = QFont('Arial', 12)
        label_font.setBold(True)
        self.label.setFont(label_font)
        self.addWidget(self.label)
        
        self.buttons = []
        self.no_lab_button = []
        
        # create no label button
        button_text = 'no label [`]'
        button = QPushButton(text=button_text)
        button.setFont(QFont('Arial', 12))
        
        button.setCheckable(True)
        button.setShortcut('`')        
        self.no_lab_button.append(button)
        self.addWidget(button)
        
        for i, lab in enumerate(self.label_names):
            if i == 0:
                button_text = f'{lab} [{i}]'
                button = QPushButton(text=button_text)
                button.setFont(QFont('Arial', 12))

                button.setCheckable(True)
                button.setShortcut(str(i))
                button.setChecked(True)  
                
            elif i < 10:
                button_text = f'{lab} [{i}]'
                button = QPushButton(text=button_text)
                button.setFont(QFont('Arial', 12))
                
                button.setCheckable(True)
                button.setShortcut(str(i))
                
            elif i >= 10:
                shortcut = self.short_lets[i-10]
                button_text = f'{lab} [{shortcut}]'
                button = QPushButton(text=button_text)
                button.setFont(QFont('Arial', 12))
                
                button.setCheckable(True)
                button.setShortcut(shortcut)
                
            rgb_val = tuple(np.round(cmap[i+1],2)*255)
            color_str = f'background-color:rgba{rgb_val}'
            button.setStyleSheet(color_str)
            self.buttons.append(button)
            self.addWidget(button)
            
        
class Automation(QVBoxLayout):
    
    def __init__(self, parent=None):
        super().__init__()
        
        self.parent = parent
        
        self.label = QLabel('Auto Behavioral Labels')
        label_font = QFont('Arial', 12)
        label_font.setBold(True)
        self.label.setFont(label_font)
        self.addWidget(self.label)

        self.loadDLC_button = QPushButton('Load DLC')
        self.loadDLC_button.setFont(QFont('Arial', 12))
        # button.setShortcut()
        # self.loadDLC_button.clicked.connect(self.getFileDLC)
        
        self.train_button = QPushButton('Add data and train model')
        self.train_button.setFont(QFont('Arial', 12))
        
        self.predict_button = QPushButton('Predict behaviors')
        self.predict_button.setFont(QFont('Arial', 12))
        
        self.copyPred_button = QPushButton('Copy predicted labels')
        self.copyPred_button.setFont(QFont('Arial', 12))
        
        self.addWidget(self.loadDLC_button)
        self.addWidget(self.train_button)
        self.addWidget(self.predict_button)
        self.addWidget(self.copyPred_button)
        
    
class GeneralButtons(QVBoxLayout):
    
    def __init__(self):
        super().__init__()
                
        self.label = QLabel('General Buttons')
        label_font = QFont('Arial', 12)
        label_font.setBold(True)
        self.label.setFont(label_font)
        self.addWidget(self.label)

        self.showFullLabels = QPushButton('Show full labels')
        self.showFullLabels.setFont(QFont('Arial', 12))
        # button.setShortcut()
        # self.loadDLC_button.clicked.connect(self.getFileDLC)
        
        self.undo_button = QPushButton('Undo labels')
        self.undo_button.setFont(QFont('Arial', 12))
        
        self.addWidget(self.showFullLabels)
        self.addWidget(self.undo_button)

          
class VideoDesc(QtWidgets.QGraphicsView):
    
    def __init__(self, parent=None,
                 video_name = [],
                 video_dims = [],
                 video_fps = [],
                 video_framecount = [],
                 num_lab_frames = []):
        
        super().__init__()
        
        self.parent = parent
        
        self.video_name = video_name
        self.video_dims = video_dims
        self.video_fps = video_fps
        self.video_framecount = video_framecount
        self.num_lab_frames = num_lab_frames
        
        self.text = QGraphicsSimpleTextItem()
        self.text.setFont(QFont('Arial', 12))
        self.updateText()
        
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.addItem(self.text)
        self.setScene(self._scene)
                  
    def updateText(self):
        self.text.setText((f'Video: {self.video_name} \n'
               f"Dimensions: {self.video_dims[0]:.0f}x{self.video_dims[1]:.0f} (W x H) \n"
               f"Frames per sec: {self.video_fps:.0f} \n"
               f"Frame count: {self.video_framecount} \n"
               f"Frames labeled: {self.num_lab_frames}" )) 
        
   
class MainWindow(QMainWindow):

    def __init__(self, screen_size):
        super().__init__()
                
        self.setStyleSheet(WIDGET_STYLES)       
        
        self.screen_size = screen_size
        
        # set important class variables
        self.frame_i = 0
        self.label_selected = np.array([0])
        self.labels = np.array([])
        self.pred_labels = np.array([])
        self.label_names = []
        # self.label_names = ['none','crick_enter','pursuit','eat']
        self.video_loaded = 0
        self.num_lab_frames = 0
        self.max_undo_num = 20
        self.prev_frame_i = []
        self.prev_leftRange = []
        self.prev_rightRange = []
        self.prev_frame_i = []
        self.prev_labels = []
        self.config = None
        
        self.ML_model = []
        self.ML_feats = np.array([])
        self.ML_labels = np.array([])
        self.ML_vid_trained = False
        self.dlc_filepath = []
        self.PCA_ON = False
        
        self.lab_buttons_layout = None
        
        self.vid_dirpath = QDir.homePath()
        self.config_dirpath = QDir.homePath()
        
        # topLayout = QGridLayout()
        topLayout = QHBoxLayout()
        
        # create videoPlayer sub widget
        self.videoPlayer, self.videoLayout = self.createVideoWidget()
        
        self.labelLayout = QVBoxLayout()
        self.labelLayout.setContentsMargins(0, 0, 0, 0)
        self.predLabelLayout = QVBoxLayout()
        self.predLabelLayout.setContentsMargins(0, 0, 0, 0)
        
        # put leftlayout into a widget in order to set the max width of the whole thing
        self.leftLayout_Wid = QWidget()
        self.leftLayout = QVBoxLayout()
        self.leftLayout_Wid.setLayout(self.leftLayout)
        self.leftLayout_Wid.setMaximumWidth(300)
        self.leftLayout_Wid.setMaximumHeight(np.round(self.screen_size.height()*0.85).astype(int))
        
        # self.rightLayout = QVBoxLayout()
        # self.rightLayout.setContentsMargins(0, 0, 0, 0)
        # self.rightLayout.addLayout(self.videoLayout)
        # self.rightLayout.addLayout(self.labelLayout)
        # self.rightLayout.addLayout(self.predLabelLayout)
                
        videoLayout_Wd = QWidget()
        videoLayout_Wd.setLayout(self.videoLayout)
        labelLayout_Wd = QWidget()
        labelLayout_Wd.setLayout(self.labelLayout)
        predLabelLayout_Wd = QWidget()
        predLabelLayout_Wd.setLayout(self.predLabelLayout)
        
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(videoLayout_Wd)
        splitter.addWidget(labelLayout_Wd)
        splitter.addWidget(predLabelLayout_Wd)
        # splitter.setOrientation(Qt.Vertical)
        
        topLayout = QSplitter()
        topLayout.addWidget(self.leftLayout_Wid)
        topLayout.addWidget(splitter)
        # topLayout.addLayout(self.rightLayout)
        
        # main_widget = QWidget()
        # main_widget.setLayout(topLayout)
        
        self.setCentralWidget(topLayout)
        
        # create all of the main window qt "actions"
        self.createActions()

        
    def goLeft(self):
        # get current slider/frame position and move video back one
        pos = self.videoPlayer.positionSlider.value()
        range_right = self.videoPlayer.right_range
        range_width = range_right - self.videoPlayer.left_range
        
        if pos < 1:
            pos = 1
            
        if pos < self.videoPlayer.left_range + 1:
            self.videoPlayer.right_range -= range_width*0.3
            self.videoPlayer.left_range -= range_width*0.3
            
        if self.videoPlayer.left_range < 1:
            self.videoPlayer.left_range = 0
            self.videoPlayer.right_range = range_right
            
        self.updateSlider()
        self.videoPlayer.positionSlider.setValue(pos-1)
        self.loadPosition(pos-1)
        
    def goRight(self):
        # get current slider/frame position and move video forward one
        pos = self.videoPlayer.positionSlider.value()
        range_left = self.videoPlayer.left_range
        range_width = self.videoPlayer.right_range - self.videoPlayer.left_range
        
        if pos > self.videoPlayer.frame_count - 2:
            pos = self.videoPlayer.frame_count - 2
            
        if pos > self.videoPlayer.right_range - 1:
            self.videoPlayer.right_range += range_width*0.3
            self.videoPlayer.left_range += range_width*0.3
            
        if self.videoPlayer.right_range > self.videoPlayer.frame_count - 2:
            self.videoPlayer.left_range = range_left
            self.videoPlayer.right_range = self.videoPlayer.frame_count - 2
            
        self.updateSlider()
        self.videoPlayer.positionSlider.setValue(pos+1)  
        self.loadPosition(pos+1)
        
    def sliderBackward(self):
        interval = self.videoPlayer.right_range - self.videoPlayer.left_range
        self.videoPlayer.left_range = int(self.videoPlayer.left_range - 0.3*interval)
        self.videoPlayer.right_range = int(self.videoPlayer.right_range - 0.3*interval)
        
        if self.videoPlayer.left_range < 0:
            self.videoPlayer.left_range = 0
            self.videoPlayer.right_range = interval
            
        if self.frame_i > self.videoPlayer.right_range:
            self.videoPlayer.right_range = self.frame_i
            
        self.updateSlider()
        
    def sliderForward(self):
        interval = self.videoPlayer.right_range - self.videoPlayer.left_range
        self.videoPlayer.left_range = int(self.videoPlayer.left_range + 0.3*interval)
        self.videoPlayer.right_range = int(self.videoPlayer.right_range + 0.3*interval)
        
        if self.videoPlayer.right_range > self.videoPlayer.frame_count:
            self.videoPlayer.left_range = self.videoPlayer.frame_count - interval
            self.videoPlayer.right_range = self.videoPlayer.frame_count
            
        if self.frame_i < self.videoPlayer.left_range:
            self.videoPlayer.left_range = self.frame_i
            
        self.updateSlider()
        
    def updateSlider(self):
        self.videoPlayer.positionSlider.setRange(int(self.videoPlayer.left_range),
                                                 int(self.videoPlayer.right_range))

        left_frame_time = frame_to_time_text(self.videoPlayer.left_range, self.videoPlayer.fps)
        self.videoPlayer.left_range_text.setText(f'Frame: {self.videoPlayer.left_range:.0f} \n'
                                                  f'{left_frame_time}')
        right_frame_time = frame_to_time_text(self.videoPlayer.right_range, self.videoPlayer.fps)
        self.videoPlayer.right_range_text.setText(f'Frame: {self.videoPlayer.right_range:.0f} \n'
                                                  f'{right_frame_time}')
        
    def sliderZoomIn(self):
        self.videoPlayer.left_range = int(np.round(np.mean([self.frame_i,self.videoPlayer.left_range])))
        right_shift = abs(self.videoPlayer.right_range - self.frame_i)*0.1
        self.videoPlayer.right_range = int(np.round(np.mean([self.frame_i,self.videoPlayer.right_range])+right_shift))
        
        if self.videoPlayer.left_range < 0:
            self.videoPlayer.left_range = 0
            
        if self.videoPlayer.right_range > self.videoPlayer.frame_count:
            self.videoPlayer.right_range = self.videoPlayer.frame_count
        
        self.updateSlider()

    def sliderZoomOut(self):
        self.videoPlayer.left_range = int(np.round(self.frame_i - (self.frame_i - self.videoPlayer.left_range)*2))
        self.videoPlayer.right_range = int(np.round(self.frame_i + (self.videoPlayer.right_range - self.frame_i)*1.9))
        
        if self.videoPlayer.left_range < 0:
            self.videoPlayer.left_range = 0
            
        if self.videoPlayer.right_range > self.videoPlayer.frame_count:
            self.videoPlayer.right_range = self.videoPlayer.frame_count
            
        self.updateSlider()
        
    def posText_update(self):
        pos_text = self.videoPlayer.posTextBox.text()
        self.loadPosition(pos_text)
    
    def keyPressEvent(self, event):
        pass
        
    def loadDLC_Press(self):
        dlc_filepath, _ = QFileDialog.getOpenFileName(self, "Load corresponding DLC file",
                                                 self.vid_dirpath)
        self.dlc_filepath = QDir.toNativeSeparators(dlc_filepath)
        
        print(f'DLC path selected: {self.dlc_filepath}')
                
    def find_dlc_file(self, vid_path, vid_name):
        
        # a little inefficient of a way to automatically search for DLC csv file
        # but sometimes it's named diferently
        
        csv_fullpath = None
        for file in os.listdir(vid_path):
            if file.endswith('.csv'):
                if ('labeled' in vid_name) or ('filtered' in vid_name):
                    file_noCsv = file.split('.')[0]
                    if (file_noCsv in vid_name) and (vid_name+'DLC' in file_noCsv):
                        print('Found DLC csv file in video directory: ')
                        csv_fullpath = os.path.join(vid_path,file)
                        print(csv_fullpath)
                        
                # elif (vid_name in file) and ('DLC' in file):
                elif (vid_name+'DLC' in file):   
                    print('Found DLC csv file in video directory: ')
                    csv_fullpath = os.path.join(vid_path,file)
                    print(csv_fullpath)
                    
        if not csv_fullpath:
            print('DLC file not found for this video')
                    
        return csv_fullpath
        
    def copyPred_Press(self):
        print('Copying automated label predictions to final labels...')
        self.labels = np.array(self.pred_labels)
        
        self.label_selected = np.array([-1])
        for b in self.buttons:
            b.setChecked(False)
                
        self.loadPosition(self.frame_i)
        
    def train_model(self):
        print('Training ML model...')
        print('Tool in development...')
        
        try:
            [self.pose_features, self.feat_meta] = FeatureExtraction.extract_features(
                self.dlc_filepath, **self.dlc_params)
            
            # remove labels and features that are not labeled yet, i.e. are -1s
            labels = np.repeat(self.labels, self.dlc_params['dlc_interp_factor'], axis=1)
            lab_inds = np.where(labels[0,:] > -1)[0] # should work to look at one label col
            labels_toAdd = np.array(labels[:,lab_inds])
            feats_toAdd = np.array(self.pose_features[:,lab_inds])
            
            # collect feats and labels in variables that grow for each video
            if not self.ML_vid_trained:
                if self.ML_feats.size > 0:
                    print('previous ML training detected')
                    self.ML_feats = np.hstack([self.ML_feats, feats_toAdd])
                    self.ML_labels = np.hstack([self.ML_labels, labels_toAdd])
                else:
                    print('first ML training')
                    self.ML_feats = feats_toAdd
                    self.ML_labels = labels_toAdd
                    
            # nTrees = 400
            verb_ON = True
            # norm_ON = False
            # self.PCA_ON = False
            fps = int(self.dlc_params['fps'])
            
            [self.ML_model, self.PCA_model] = TrainModel_RF.train_model_RF(
                self.ML_feats.T, self.ML_labels.T, verb_ON, fps, **self.ML_params)
        
            self.ML_vid_trained = True
            
        except Exception as e:
            print('Error running automated behavior detection:')
            print(e)
            
        
    def model_predict(self):
        # dlc_filepath, _ = QFileDialog.getOpenFileName(self, "Load corresponding DLC file",
        #                                          self.vid_dirpath)
        # self.dlc_filepath = QDir.toNativeSeparators(dlc_filepath)        
        
        if self.ML_model and self.dlc_filepath:
                        
            [self.pose_features, self.feat_meta] = FeatureExtraction.extract_features(
                self.dlc_filepath, **self.dlc_params)
            
            self.pose_features[np.isnan(self.pose_features)] = 0
            
            if self.PCA_ON:
                print('Running PCA on features for prediction...')
                self.pose_features = self.PCA_model.transform(self.pose_features.T)
            else:
                self.pose_features = self.pose_features.T
                        
            if not self.ML_params['double_train_ON']:
                self.ML_model.verbose = 1
                pred_lab_probs = self.ML_model.predict_proba(self.pose_features)
                    
                # convert predicted label probs to array
                pred_prob_array = np.zeros(shape=(self.labels.shape)).T
                for i in range(len(pred_lab_probs)):
                    lab_probs = pred_lab_probs[i]
    
                    up_factor = int(lab_probs.shape[0]/self.labels.shape[1])
                    if up_factor > 1:
                        lab_probs = lab_probs[np.arange(lab_probs.shape[0],step=up_factor),:]
                        
                    if lab_probs.shape[1] > 1:
                        pred_prob_array[:,i] = lab_probs[:,1]
                    else:
                        pred_prob_array[:,i] = np.zeros(shape=(lab_probs.shape[0]))
                
                # select highest prob label if none chosen for a frame
                pred_labels = np.zeros(shape=pred_prob_array.shape)
                pred_labels[pred_prob_array > 0.5] = 1
                for i in range(pred_labels.shape[0]):
                    no_lab = (pred_labels[i,:] == 1).any()
                    
                    if no_lab == False:
                        max_lab = np.argmax(pred_prob_array[i,:])
                        pred_labels[i,max_lab] = 1                
                
            else:
                self.ML_model[0].verbose = 1
                self.ML_model[1].verbose = 1
                fps = int(self.dlc_params['fps'])
                
                pred_lab_probs = self.ML_model[0].predict_proba(self.pose_features)
                 
                # convert predicted label probs to array
                pred_prob_array = np.zeros(shape=(self.pose_features.shape[0],
                                                  self.labels.shape[0]))
                for i in range(len(pred_lab_probs)):
                    lab_probs = pred_lab_probs[i]
                        
                    if lab_probs.shape[1] > 1:
                        pred_prob_array[:,i] = lab_probs[:,1]
                    else:
                        pred_prob_array[:,i] = np.zeros(shape=(lab_probs.shape[0]))
                
                # select highest prob label if none chosen for a frame
                pred_labels = np.zeros(shape=pred_prob_array.shape)
                pred_labels[pred_prob_array > 0.5] = 1
                for i in range(pred_labels.shape[0]):
                    no_lab = (pred_labels[i,:] == 1).any()
                    
                    if no_lab == False:
                        max_lab = np.argmax(pred_prob_array[i,:])
                        pred_labels[i,max_lab] = 1  

                # create pred_label_features
                # diff_inds = [-15,-8,-4,-3,-2,-1,1,2,3,10]
                diff_inds = [-10*fps,-5*fps,-1*fps,-0.25*fps,-0.15*fps,-0.1*fps,-1,
                             1,0.1*fps,0.15*fps,1*fps,5*fps]
                diff_inds = np.ceil(diff_inds).astype(int) 
                
                input_feats = pred_labels.T
                label_feats = np.zeros(shape=(input_feats.shape[0]*len(diff_inds),
                                              input_feats.shape[1]))
                for ind, i in enumerate(diff_inds):
                    inds = np.arange(ind*input_feats.shape[0],
                                     ind*input_feats.shape[0]+input_feats.shape[0])
                    if i < 0:
                        diff = input_feats[:,:i]
                        label_feats[inds,:] = np.hstack([np.zeros(shape=(input_feats.shape[0],-i)), diff])
                    else:
                        diff = input_feats[:,i:]
                        label_feats[inds,:] = np.hstack([diff, np.zeros(shape=(input_feats.shape[0],i))])   
                        
                full_feats = np.hstack([label_feats.T,self.pose_features])
                
                pred_lab_probs = self.ML_model[1].predict_proba(full_feats)
                
                # convert predicted label probs to array
                pred_prob_array = np.zeros(shape=(self.labels.shape)).T
                for i in range(len(pred_lab_probs)):
                    lab_probs = pred_lab_probs[i]
    
                    up_factor = int(lab_probs.shape[0]/self.labels.shape[1])
                    if up_factor > 1:
                        lab_probs = lab_probs[np.arange(lab_probs.shape[0],step=up_factor),:]
                        
                    if lab_probs.shape[1] > 1:
                        pred_prob_array[:,i] = lab_probs[:,1]
                    else:
                        pred_prob_array[:,i] = np.zeros(shape=(lab_probs.shape[0]))
                
                # select highest prob label if none chosen for a frame
                pred_labels = np.zeros(shape=pred_prob_array.shape)
                pred_labels[pred_prob_array > 0.5] = 1
                for i in range(pred_labels.shape[0]):
                    no_lab = (pred_labels[i,:] == 1).any()
                    
                    if no_lab == False:
                        max_lab = np.argmax(pred_prob_array[i,:])
                        pred_labels[i,max_lab] = 1              
            
            # temp plot model label probs
            fig = plt.figure()
            ax = plt.gca()
            for i in range(len(pred_lab_probs)):
                
                if pred_lab_probs[i].shape[1] > 1:
                    plt.plot(pred_lab_probs[i][:,1],color=self.c_map[i+1])
                    
            
            self.pred_labels = pred_labels.T
            
            self.predlabel_full.view_labels = self.pred_labels
            self.predlabel_plot.view_labels = self.pred_labels
            
            self.predlabel_full.update_image()
            self.predlabel_plot.update_image()

        
    def labelButtonPress(self):
        # this handy built in function returns which button connected to this function was pressed
        buttonPressed = self.sender()
        
        # my code logic to achieve desired effect is clunky
        if buttonPressed is self.no_lab_button:
            if not self.no_lab_button.isChecked():
                
                button_states = []
                for b in self.buttons:
                    button_states.append(b.isChecked())
                
                # transfer button states to label selected for implementation
                if True in button_states:
                    self.label_selected = np.where(button_states)[0]
                else:
                    self.label_selected = np.array([-1])
                    
            else:
                self.label_selected = np.array([-1])
            
        else:
            # unpress no lab button if another label button is clicked
            self.no_lab_button.setChecked(False)
            
            prev_button_states = []
            for b in self.buttons:
                prev_button_states.append(b.isChecked())       
                
            button_states = []
            for b in self.buttons:
                button_states.append(b.isChecked())
        
            # if no buttons are checked check the no behavior button unless that was button pressed
            if True not in button_states and buttonPressed is not self.buttons[0]:
                self.buttons[0].setChecked(True)
                button_states[0] = True
                
            # # uncheck no behavior if clicked and already pressed
            if True in prev_button_states[1:]:
                self.buttons[0].setChecked(False)
                button_states[0] = False
                
            # transfer button states to label selected for implementation
            prev_lab_sel = self.label_selected
            self.label_selected = np.where(button_states)[0]
            
            if not np.array_equal(prev_lab_sel,self.label_selected):
                self.storePrevLabels()
                
        self.loadPosition(self.frame_i)
        
    def storePrevLabels(self):
        
        # remove first entry in stored prev labels if above max number of undos
        if len(self.prev_frame_i) > self.max_undo_num:
            self.prev_frame_i.pop(0)
            self.prev_leftRange.pop(0)
            self.prev_rightRange.pop(0)
            self.prev_labels.pop(0)    
            
        # append current label variables in prev stores for undo button
        self.prev_frame_i.append(self.frame_i)
        self.prev_leftRange.append(self.videoPlayer.left_range)
        self.prev_rightRange.append(self.videoPlayer.right_range)
        self.prev_labels.append(np.array(self.labels))
        
    def undo_Press(self):
        
        # if there are prev stores then apply them and remove last entry
        if len(self.prev_frame_i) > 0:
            self.labels = np.array(self.prev_labels[-1])
            self.frame_i = self.prev_frame_i[-1]
            self.videoPlayer.left_range = self.prev_leftRange[-1]
            self.videoPlayer.right_range = self.prev_rightRange[-1]
            
            self.prev_frame_i.pop(-1)
            self.prev_leftRange.pop(-1)
            self.prev_rightRange.pop(-1)
            self.prev_labels.pop(-1)          
        
        self.loadPosition(self.frame_i)
        
    def showFullLabel_Press(self):
        
        if self.c_map.size:
            last_frame_labeled = np.max(np.argwhere(np.array(self.labels) != -1)) # find last instance of -1 (unlabeled frame)
            xlims = [0, last_frame_labeled]
            plot_full_labels(self.labels,self.c_map, xlims)
    
    def loadPosition(self, position):
                
        if position is not None and position != '':
            position = int(position)
            if position < 0:
                position = 0
            elif position > self.videoPlayer.frame_count - 1:
                position = self.videoPlayer.frame_count - 1
                
            if self.video_loaded:    
                if self.frame_i != position:
                    self.videoPlayer.displayFrame(position)
                    
                self.updateLabels(position)
                self.frame_i = position
                self.videoPlayer.frame_i = position
                
                self.label_full.update_image()
                self.label_plot.update_image()

                self.predlabel_full.update_image()
                self.predlabel_plot.update_image()
                                
                self.videoPlayer.positionSlider.setValue(position)
                self.videoPlayer.posTextBox.setText(str(position))
            
                self.updateVideoDesc()
            
    def updateLabels(self, position):
        
        # longwinded way of making sure all the label plots are on the same page
        self.label_full.frame_i = position        
        self.label_plot.frame_i = position
        self.predlabel_full.frame_i = position
        self.predlabel_plot.frame_i = position
        
        label_range = np.arange(len(self.label_names))
        if -1 not in self.label_selected:
            if position > self.frame_i:
                
                for l in label_range:
                    if l in self.label_selected:
                        self.labels[l,self.frame_i:position+1] = 1
                    else:
                        self.labels[l,self.frame_i:position+1] = 0
                    
            elif position < self.frame_i:

                for l in label_range:
                    if l in self.label_selected:
                        self.labels[l,position:self.frame_i] = 1
                    else:
                        self.labels[l,position:self.frame_i] = 0                    
            else:

                for l in label_range:
                    if l in self.label_selected:
                        self.labels[l,self.frame_i] = 1
                    else:
                        self.labels[l,self.frame_i] = 0
                        
        self.label_full.view_labels = self.labels
        self.label_plot.view_labels = self.labels
        self.num_lab_frames = np.sum(self.labels[0,:] > -1)
        
    def createColorMap(self):
        num_labels = len(self.label_names) - 1
        # uk_labels = list(np.arange(-1,num_labels+1)) # assumes labels are ascending ints
        
        R = np.linspace(0, 1, num_labels)
        cmap = plt.cm.get_cmap("Spectral")(R)
        cmap = cmap*0.85 # Make the colors darker
        no_beh_color = [0.8, 0.8, 0.8, 1]
        not_coded_color = [1,1,1,1]
        cmap = np.insert(cmap, 0, no_beh_color, axis=0)
        cmap = np.insert(cmap, 0, not_coded_color, axis=0)
        
        return cmap
    
    def newLabels(self, frame_count):
        self.labels = -np.ones(shape=(len(self.label_names), frame_count))
        self.pred_labels = -np.ones(shape=(len(self.label_names), frame_count))
        self.frame_i = 0
        
        if self.video_loaded:
            for i in reversed(range(self.labelLayout.count())): 
                self.labelLayout.itemAt(i).widget().setParent(None)
            for i in reversed(range(self.predLabelLayout.count())): 
                self.predLabelLayout.itemAt(i).widget().setParent(None)                
            
        num_labels = len(self.label_names) - 1
        self.c_map = self.createColorMap()
        self.label_full = FullLabelViewer(parent=[], labels=self.labels,
                                      num_labels=num_labels, c_map = self.c_map)
        self.label_plot = LabelViewer(parent=[], labels=self.labels,
                                      num_labels=num_labels, c_map = self.c_map,
                                      fps=self.videoPlayer.fps)
        label_text = QLabel('Final labels')
        label_text.setFont(QFont('Arial', 12))
        self.labelLayout.addWidget(label_text)
        self.labelLayout.addWidget(self.label_full)
        self.labelLayout.addWidget(self.label_plot)
        
        self.predlabel_full = FullLabelViewer(parent=[], labels=self.pred_labels,
                                      num_labels=num_labels, c_map = self.c_map)
        self.predlabel_plot = LabelViewer(parent=[], labels=self.pred_labels,
                                          num_labels=num_labels, c_map = self.c_map,
                                          fps=self.videoPlayer.fps)
        predlabel_text = QLabel('Predicted labels')
        predlabel_text.setFont(QFont('Arial', 12))
        self.predLabelLayout.addWidget(predlabel_text)
        self.predLabelLayout.addWidget(self.predlabel_full)
        self.predLabelLayout.addWidget(self.predlabel_plot) 
        self.loadPosition(0)

    def createVideoWidget(self):
        videoPlayer = VideoWindow(parent=self)
        videoPlayer.show()
        videoPlayer.positionSlider.sliderMoved.connect(self.loadPosition)
        videoPlayer.posTextBox.editingFinished.connect(self.posText_update)        
        videoPlayer.slider_backward_but.clicked.connect(self.sliderBackward)
        videoPlayer.slider_forward_but.clicked.connect(self.sliderForward)
        videoPlayer.slider_minus_but.clicked.connect(self.sliderZoomOut)
        videoPlayer.slider_plus_but.clicked.connect(self.sliderZoomIn)
        videoPlayer.left_range_text = QLabel('0')
        videoPlayer.right_range_text = QLabel('0')
        
        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(2, 2, 2, 2)
        # controlLayout.addWidget(self.videoPlayer.playButton)
        controlLayout.addWidget(videoPlayer.left_range_text)
        controlLayout.addWidget(videoPlayer.positionSlider)
        controlLayout.addWidget(videoPlayer.right_range_text)
        controlLayout.addWidget(videoPlayer.slider_backward_but)
        controlLayout.addWidget(videoPlayer.slider_forward_but)
        controlLayout.addWidget(videoPlayer.slider_minus_but)
        controlLayout.addWidget(videoPlayer.slider_plus_but)
        controlLayout.addWidget(videoPlayer.posTextBox)
        
        # create a vertical box layout and add the two labels
        videoLayout = QVBoxLayout()
        videoLayout.setContentsMargins(0, 0, 0, 0)
        # videoLayout.addWidget(self.videoPlayer.image_label)
        videoLayout.addWidget(videoPlayer)
        # videoLayout.addWidget(videoPlayer.textLabel)
        videoLayout.addLayout(controlLayout)

        return (videoPlayer, videoLayout)        
    
    def loadVideo(self, filepath):
        self.vid_filepath = QDir.toNativeSeparators(filepath)
        self.videoPlayer.openFile(self.vid_filepath)
        
        path_split =  self.vid_filepath.split(os.sep)
        self.vid_dirpath = (os.sep).join(path_split[:-1])
        self.vid_name = path_split[-1].split('.')[0] # get video filename without file ending type

        if not self.video_loaded:
            self.vid_label = VideoDesc(parent=self,
                                                video_name=self.vid_name,
                                                video_dims=[self.videoPlayer.disply_width,
                                                            self.videoPlayer.display_height],
                                                video_fps=self.videoPlayer.fps,
                                                video_framecount=self.videoPlayer.frame_count,
                                                num_lab_frames=0)
            # create section title
            label = QLabel('Video Description')
            label_font = QFont('Arial', 12)
            label_font.setBold(True)
            label.setFont(label_font)
            self.leftLayout.addWidget(label)
            self.leftLayout.addWidget(self.vid_label)
            
            # text input dialog window for defining labels
            if DEBUG_ON:
                print('Skipping labels entry...')
                print('Labels are: ')              
                print(self.label_names)
                
            elif self.config is None:
                # self.defineLabels()
                self.getConfig()
            
            self.newLabels(self.videoPlayer.frame_count)
            self.loadButtons()
            
        else:
            self.updateVideoDesc()
            self.newLabels(self.videoPlayer.frame_count)
            
        self.updateSlider()      
                
        if self.ML_model:
            self.dlc_filepath = self.find_dlc_file(self.vid_dirpath, self.vid_name)
            self.model_predict()
                
        self.video_loaded = 1
        self.ML_vid_trained = False
            
    def updateVideoDesc(self):
        self.vid_label.video_name = self.vid_name
        self.vid_label.video_dims = [self.videoPlayer.disply_width,
                                     self.videoPlayer.display_height]
        self.vid_label.video_fps = self.videoPlayer.fps
        self.vid_label.video_framecount = self.videoPlayer.frame_count
        self.vid_label.num_lab_frames = self.num_lab_frames
        
        self.vid_label.updateText()
        
    def loadButtons(self):
        genButtonLayout = GeneralButtons()
        genButtonLayout.showFullLabels.clicked.connect(self.showFullLabel_Press)
        genButtonLayout.undo_button.clicked.connect(self.undo_Press)
        self.leftLayout.addLayout(genButtonLayout)
        
        automationLayout = Automation(parent=self)
        automationLayout.loadDLC_button.clicked.connect(self.loadDLC_Press)
        automationLayout.train_button.clicked.connect(self.train_model)
        automationLayout.predict_button.clicked.connect(self.model_predict)
        automationLayout.copyPred_button.clicked.connect(self.copyPred_Press)
        self.leftLayout.addLayout(automationLayout)
        
        self.loadLabelButtons()
        
    def loadLabelButtons(self):
        if self.lab_buttons_layout is not None:
            print('previous button layout detected')
            self.lab_buttons_layout.setParent(None)

        buttonsLayout = LabelButtons(parent=self, label_names=self.label_names, cmap=self.c_map)
        
        buttonsLayout_Ht = QWidget()
        buttonsLayout_Ht.setLayout(buttonsLayout)
        buttonsLayout_Ht.setMaximumHeight(500)        
        
        self.no_lab_button = buttonsLayout.no_lab_button[0]
        self.buttons = buttonsLayout.buttons
        self.lab_buttons_layout = buttonsLayout_Ht
        
        self.no_lab_button.clicked.connect(self.labelButtonPress)
        for b in self.buttons:
            b.clicked.connect(self.labelButtonPress)

        self.leftLayout.addWidget(buttonsLayout_Ht)
               
    def getFile(self):
        self.vid_filepath, _ = QFileDialog.getOpenFileName(self, "Open video",
                                                  self.vid_dirpath)
        
        self.loadVideo(self.vid_filepath)
        
    # def defineLabels(self):
    #     text, ok = QInputDialog.getText(self, 'Label input',
    #                                    'list your behavior labels (e.g.: eat, drink, sleep):')
    #     text = text.split(',')
    #     text.insert(0,'none')
    #     if ok:
    #         print('Labels selected:')
    #         print(text)
    #         self.label_names = text
            
    def getConfig(self):
        # use filedialog to manually find path to config.yaml file for this video
        # project and load its contents into class variable
        file_name = QFileDialog.getOpenFileName(self, 'Select config(.yaml) file to open...',
                                               self.vid_dirpath)
        file_path = QDir.toNativeSeparators(file_name[0])
        # self.config_dirpath = os.path.join(file_path.split(os.path.sep)[:-1])
        # print(self.config_dirpath)
        
        self.loadConfig(file_path)
        
        print(f'Config file loaded: {file_path}')

        
    def loadConfig(self,config_path):
        
        # parses yaml file into self.config dictionary
        self.config = open_config(config_path)
        self.label_names = self.config['labels']
        self.videoPlayer.fps = self.config['fps']
        
        self.dlc_params = {
            'xlims': self.config['xlims'],
            'ylims': self.config['ylims'],
            'fps': self.config['fps'],
            'dlc_conf_thresh': self.config['dlc_conf_thresh'],
            'dlc_interp_factor': self.config['dlc_interp_factor'],
            'interp_fill_lim': self.config['interp_fill_lim'],
            'dlc_smooth_ON': self.config['dlc_smooth_ON'],
            'dlc_sm_sec': self.config['dlc_sm_sec'],
            'no_abs_pose': self.config['no_abs_pose'],
            'pose_to_exclude': self.config['pose_to_exclude'],
            'head_body_norm': self.config['head_body_norm']
            }
        
        print(self.dlc_params)
        
        self.ML_params = {
            'max_model_len': self.config['max_model_len'],
            'model_nTrees': self.config['model_nTrees'],
            'ML_norm_ON': self.config['ML_norm_ON'],
            'ML_PCA_ON': self.config['ML_PCA_ON'],
            'double_train_ON': self.config['double_train_ON']}
        
        
        if self.video_loaded:
            self.newLabels(self.videoPlayer.frame_count)
            self.loadLabelButtons()
            
        self.loadPosition(self.frame_i)

    def openLabels(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select (_LabeledInds) file to open...',
                                               self.vid_dirpath)
        file_path = QDir.toNativeSeparators(file_name[0])
        print('Opening Label Data...')
           
        with open(file_path, newline = '') as file:
            output = np.loadtxt(file, delimiter=',', skiprows=0, dtype=str)
        
        file_labels_names = output[0,1:]
        file_labels = np.array(output[1:,1:].astype(int).T) # get rid of first frame col and transpose
        
        if len(file_labels_names) > len(self.label_names):
            print('Imported save file has additional labels:')
            print(f'Defined {len(self.label_names)} labels, save file has: {len(file_labels_names)}')
            print('Using save file label names...')
            self.labels = np.array(file_labels)
            self.label_names = file_labels_names
            
            if self.video_loaded:
                self.newLabels(self.videoPlayer.frame_count)
                self.loadLabelButtons()            
        elif len(file_labels_names) < len(self.label_names):
            print('Imported save file has fewer labels:')
            print(f'Defined {len(self.label_names)} labels, save file has: {len(file_labels_names)}')
            print('Using defined label names...') 
            self.labels = np.zeros(shape=(len(self.label_names),
                                          file_labels.shape[1]))
            self.labels[np.arange(len(file_labels_names)),:] = file_labels
            
            if self.video_loaded:
                self.newLabels(self.videoPlayer.frame_count)
                self.loadLabelButtons()  
        # else:
        #     self.labels = np.array(file_labels)
                
        # click "no label" button and set all other buttons to unchecked
        self.label_selected = np.array([-1])
        self.no_lab_button.setChecked(True)
        for b in self.buttons:
            b.setChecked(False)     
            
        self.labels = np.array(file_labels)
        self.loadPosition(self.frame_i)
        
    def saveLabels(self):
        save_dir = QFileDialog.getExistingDirectory(self, 'Select save location for labels...',
                                               self.vid_dirpath)
        save_dir = QDir.toNativeSeparators(save_dir)
        print('Saving Label Data as...')
        
        labels_toSave = np.array(self.labels)
        vid_path = self.vid_filepath.split(os.sep)
        vid_name = vid_path[-1].split('.')[0] # get video filename without file ending type
        save_path_inds = os.path.join(save_dir, vid_name + '_LabeledInds' + '.csv')
        print(save_path_inds)
        with open(save_path_inds, 'w', newline='') as f:
            writer = csv.writer(f)
            header = self.label_names[:] # make new list variable
            header.insert(0, 'frame')
            writer.writerow(header)
            
            # write data
            for i, l in enumerate(labels_toSave.T):
                row = [int(x) for x in l]
                row.insert(0,i+1) # start the frame count at 1 not 0
                                
                writer.writerow(row)
        
        # find inds where there are changes in list elements
        # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
        bout_change_inds = []
        for ind, l in enumerate(labels_toSave):
            l[np.where(l==-1)] = 0 # remove -1s from label vector to only look at labeled bouts
            l = np.append(l,0) # add 0 to end to allow last labeled frame to be caught by roll
            rolled_labels = np.roll(l, 1)
            rolled_labels[0] = 0
            bout_change_inds.append(np.where(rolled_labels!=l))        
        
        # Create lists for start and end indices of each bout
        bout_st_inds = []
        bout_end_inds = []
        bout_labels = []
        for l in range(len(bout_change_inds)):
            label_bout_inds = bout_change_inds[l][0]
            
            for i in np.arange(len(label_bout_inds),step=2):
                ind_i = label_bout_inds[i]
                next_ind_i = label_bout_inds[i+1]
                label_l = l
                
                bout_st_inds.append(ind_i+1)
                bout_end_inds.append(next_ind_i)
                bout_labels.append(label_l)
        
        sort_ind = np.argsort(bout_st_inds).astype(int)
        
        bout_st_inds = [bout_st_inds[x] for x in sort_ind]
        bout_end_inds = [bout_end_inds[x] for x in sort_ind]
        bout_labels = [bout_labels[x] for x in sort_ind]
        
        save_path_bouts = os.path.join(save_dir, vid_name + '_LabeledBouts' + '.csv')
        print(save_path_bouts)
        with open(save_path_bouts, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['start_frame','end_frame','label']
            writer.writerow(header)
            
            # write data
            for i, j, l in zip(bout_st_inds, bout_end_inds, bout_labels):
                row = [i, j, l]
                writer.writerow(row)                
                 
        print('Labels Saved')
                
    def saveModel(self):
        save_dir = QFileDialog.getExistingDirectory(self, 'Select save location for ML model...',
                                               self.vid_dirpath)
        save_dir = QDir.toNativeSeparators(save_dir)        

        vid_path = self.vid_filepath.split(os.sep)
        vid_name = vid_path[-1].split('.')[0] # get video filename without file ending type
        time = datetime.today().strftime('%Y-%m-%d_H%H-M%M')
                                         
        filename = vid_name + '_' + time + '_RF_Mdl.pkl'
        pkl_filename = os.path.join(save_dir,filename)
        
        print('Saving ML model...')
        print(pkl_filename)
        
        with open(pkl_filename,'wb') as pkl_out:
         	pickle.dump([self.ML_model, self.PCA_model],pkl_out,-1)
             
        print('Finished!')
        
    def saveFeatData(self):
        save_dir = QFileDialog.getExistingDirectory(self, 'Select save location for feat data...',
                                               self.vid_dirpath)
        save_dir = QDir.toNativeSeparators(save_dir)        

        vid_path = self.vid_filepath.split(os.sep)
        vid_name = vid_path[-1].split('.')[0] # get video filename without file ending type
        time = datetime.today().strftime('%Y-%m-%d_H%H-M%M')
                                         
        filename = vid_name + '_' + time + '_feat_data.pkl'
        pkl_filename = os.path.join(save_dir,filename)
        
        print('Saving feat data...')
        print(pkl_filename)
          
        with open(pkl_filename,'wb') as pkl_out:
         	pickle.dump([self.ML_feats , self.ML_labels],pkl_out,-1)
             
        print('Finished!')        
        
    def openModel(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select location of ML model .pkl file...',
                                               self.vid_dirpath)
        file_path = QDir.toNativeSeparators(file_name[0])
    
        print('Opening ML model...')
        print(file_path)
        
        # load model from pickle file
        with open(file_path,'rb') as pkl_in:
        	file_output = pickle.load(pkl_in)

        if file_output[1]:
            self.PCA_ON = True
            [self.ML_model, self.PCA_model] = file_output
        elif not file_output[1]:
            self.PCA_ON = False
            [self.ML_model, self.PCA_model] = file_output   
            
        print('Model Loaded!')
                
    def openFeatData(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select location of feat data .pkl file...',
                                               self.vid_dirpath)
        file_path = QDir.toNativeSeparators(file_name[0])
    
        print('Opening feat data...')
        print(file_path)
        
        # load feat data from pickle file
        with open(file_path,'rb') as pkl_in:
        	file_output = pickle.load(pkl_in)

        [self.ML_feats, self.ML_labels] = file_output
            
        print('Data Loaded!')        
        
    def createActions(self):
        
        # Create open video file action
        openVideoAction = QAction(QIcon('open.png'), '&Open Video', self)        
        openVideoAction.setShortcut('Ctrl+O')
        openVideoAction.setStatusTip('Open movie')
        openVideoAction.triggered.connect(self.getFile)
        
        # Create open label action
        openConfigAction = QAction(QIcon(''), '&Open Config File', self)        
        openConfigAction.setShortcut('Ctrl+F')
        openConfigAction.setStatusTip('Open config .yaml file')
        openConfigAction.triggered.connect(self.getConfig)
        
        # Create open label action
        openLabelAction = QAction(QIcon(''), '&Open Labels', self)        
        openLabelAction.setShortcut('Ctrl+L')
        openLabelAction.setStatusTip('Open labels from csv save file')
        openLabelAction.triggered.connect(self.openLabels)
        
        # Create save label action
        saveLabelAction = QAction(QIcon(''), '&Save Labels', self)        
        saveLabelAction.setShortcut('Ctrl+S')
        saveLabelAction.setStatusTip('Save labels to file')
        saveLabelAction.triggered.connect(self.saveLabels)
        
        # Create open model action
        openModelAction = QAction(QIcon(''), '&Open ML Model', self)        
        openModelAction.setShortcut('Ctrl+V')
        openModelAction.setStatusTip('Open machine learning model from .pkl file')
        openModelAction.triggered.connect(self.openModel)
        
        # Create save model action
        saveModelAction = QAction(QIcon(''), '&Save ML Model', self)        
        saveModelAction.setShortcut('Ctrl+C')
        saveModelAction.setStatusTip('Save machine learning model to file')
        saveModelAction.triggered.connect(self.saveModel)
        
        # Create open feat data action
        openDataAction = QAction(QIcon(''), '&Open Feature Data', self)        
        # openModelAction.setShortcut('Ctrl+V')
        openDataAction.setStatusTip('Open feature data for ML learning from .pkl file')
        openDataAction.triggered.connect(self.openFeatData)
        
        # Create save feat data action
        saveDataAction = QAction(QIcon(''), '&Save Feature Data', self)        
        # saveModelAction.setShortcut('Ctrl+C')
        saveDataAction.setStatusTip('Save feature data to file')
        saveDataAction.triggered.connect(self.saveFeatData)        
        
        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        menuBar.setFont(QFont('Arial',16))
        
        actions = [openVideoAction, openConfigAction, openLabelAction, saveLabelAction,
                   openModelAction, saveModelAction, openDataAction, saveDataAction, exitAction]
        
        for a in actions:
            a.setFont(QFont('Arial',12))
            fileMenu.addAction(a)
            
        # make shortcut qactions for mainwindow to avoid children from capturing keyevents
        goLeftAction = QAction(self)
        goLeftAction.setShortcut('left')
        goLeftAction.triggered.connect(self.goLeft)
        self.addAction(goLeftAction)
        
        goRightAction = QAction(self)
        goRightAction.setShortcut('right')
        goRightAction.triggered.connect(self.goRight)
        self.addAction(goRightAction)            
        
    def resizeEvent(self, event):
        self.loadPosition(self.frame_i)
    
    def exitCall(self):
        sys.exit(app.exec_())
       
    
    


if __name__ == '__main__':
    
    plt.ion()
    plt.close(1)
    plt.close(2)
    
    app = QApplication.instance()
    if app == None:
        app = QApplication([])
    
    screen = app.primaryScreen()
    screen_size = screen.size()
    
    window = MainWindow(screen_size)
    window.resize(1000, 700)
    window.show()
    
    print('Load config .yaml file to get started')
    
    if DEBUG_ON:

        window.loadConfig(conf_path)
        window.loadVideo(vid_path)
        # window.dlc_filepath = dlc_filepath
        # window.openLabels(lab_path)

    app.exec_()
        








