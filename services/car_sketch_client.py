import os
import sys
import cv2
import time
import grpc
import numpy as np

from PIL import Image
from PyQt5.QtCore import *  # Qt
from PyQt5.QtGui import *  # QPainter, QPainterPath
from PyQt5.QtWidgets import *  # QWidget, QApplication

from services import image_transfer_pb2
from services import image_transfer_pb2_grpc

from util import util
from ui_shadow_draw.ui_sketch import UISketch
from ui_shadow_draw.gangate_vis import GANGATEVis
from ui_shadow_draw.ui_recorder import UIRecorder
from ui_shadow_draw.gangate_draw import GANGATEDraw


channel = grpc.insecure_channel('192.168.1.131:50066')
stub = image_transfer_pb2_grpc.ImageTransferStub(channel)


def get_Image(image, label):
    return image_transfer_pb2.Image(data=image, label=label)


def call_GetGenerateImage(image_np, label=0):
    response = stub.GetGenerateImage(get_Image(image_np.tobytes(), label))
    return response


def call_RandomizeNoise():
    stub.RandomizeNoise(image_transfer_pb2.Empty())


class GANGATEGui(QWidget):
    def __init__(self, win_size=256, img_size=256):
        QWidget.__init__(self)

        self.win_size = win_size
        self.img_size = img_size
        self.num_interpolate = 6

        self.drawWidget = GANGATEDraw(
            win_size=self.win_size, img_size=self.img_size)
        self.drawWidget.setFixedSize(win_size, win_size)

        self.visWidget = GANGATEVis(
            win_size=self.win_size, img_size=self.img_size)
        self.visWidget.setFixedSize(win_size, win_size)

        vbox = QVBoxLayout()

        self.drawWidgetBox = QGroupBox()
        self.drawWidgetBox.setTitle('Drawing Pad')
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(self.drawWidget)
        self.drawWidgetBox.setLayout(vbox_t)
        vbox.addWidget(self.drawWidgetBox)

        self.labelId = 0

        self.bGenerate = QPushButton('Generate')
        self.bGenerate.setToolTip(
            "This button generates the final image to render")

        self.bReset = QPushButton('Reset')
        self.bReset.setToolTip("This button resets the drawing pad !")

        self.bRandomize = QPushButton('Dice')
        self.bRandomize.setToolTip(
            "This button generates new set of generations the drawing pad !")

        self.bMoveStroke = QRadioButton('Move Stroke')
        self.bMoveStroke.setToolTip("This button resets the drawing pad !")

        self.bWarpStroke = QRadioButton('Warp Stroke')
        self.bWarpStroke.setToolTip("This button resets the drawing pad !")

        self.bDrawStroke = QRadioButton('Draw Stroke')
        self.bDrawStroke.setToolTip("This button resets the drawing pad !")

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)

        vbox3 = QVBoxLayout()
        self.visWidgetBox = QGroupBox()
        self.visWidgetBox.setTitle('Generations')

        vbox_t3 = QVBoxLayout()
        vbox_t3.addWidget(self.visWidget)
        self.visWidgetBox.setLayout(vbox_t3)
        vbox3.addWidget(self.visWidgetBox)

        bhbox_controls = QGridLayout()
        bGroup_controls = QButtonGroup(self)

        bGroup_controls.addButton(self.bReset)
        bGroup_controls.addButton(self.bDrawStroke)
        bGroup_controls.addButton(self.bMoveStroke)
        bGroup_controls.addButton(self.bWarpStroke)

        bhbox_controls.addWidget(self.bReset, 0, 0)
        bhbox_controls.addWidget(self.bRandomize, 0, 1)
        bhbox_controls.addWidget(self.bDrawStroke, 0, 2)
        bhbox_controls.addWidget(self.bMoveStroke, 0, 3)
        bhbox_controls.addWidget(self.bWarpStroke, 0, 4)

        hbox.addLayout(vbox3)

        controlBox = QGroupBox()
        controlBox.setTitle('Controls')

        controlBox.setLayout(bhbox_controls)

        vbox_final = QVBoxLayout()
        vbox_final.addLayout(hbox)
        vbox_final.addWidget(controlBox)
        self.setLayout(vbox_final)

        self.bDrawStroke.setChecked(True)

        self.bGenerate.clicked.connect(self.generate)
        self.bReset.clicked.connect(self.reset)
        self.bRandomize.clicked.connect(self.randomize)
        self.bMoveStroke.clicked.connect(self.move_stroke)
        self.bWarpStroke.clicked.connect(self.warp_stroke)
        self.bDrawStroke.clicked.connect(self.draw_stroke)

    def browse(self, pos_y, pos_x):
        try:
            num_rows = int(self.num_interpolate / 2)
            num_cols = 2
            div_rows = int(self.img_size / num_rows)
            div_cols = int(self.img_size / num_cols)

            which_row = int(pos_x / div_rows)
            which_col = int(pos_y / div_cols)

            cv2_gallery = cv2.imread('imgs/fake_B_gallery.png')
            cv2_gallery = cv2.resize(
                cv2_gallery, (self.img_size, self.img_size))

            cv2_gallery = cv2.rectangle(
                cv2_gallery, (which_col * div_cols, which_row * div_rows), ((
                    which_col + 1) * div_cols, (which_row + 1) * div_rows),
                (0, 255, 0), 8)
            self.visWidget.update_vis_cv2(cv2_gallery)

            cv2_img = cv2.imread('imgs/test_fake_B_shadow.png')
            cv2_img = cv2.resize(cv2_img, (self.img_size, self.img_size))

            which_highlight = which_row * 2 + which_col
            img_gray = cv2.imread('imgs/test_%d_L_fake_B_inter.png' %
                                  (which_highlight), cv2.IMREAD_GRAYSCALE)
            img_gray = cv2.resize(img_gray, (self.img_size, self.img_size))
            (thresh, im_bw) = cv2.threshold(
                img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2_img[np.where(im_bw == [0])] = [0, 255, 0]

            self.drawWidget.setShadowImage(cv2_img)
        except Exception as e:
            print(e)

    def generate(self):
        cv2_scribble = self.drawWidget.getDrawImage()  # 256x256x3
        response = call_GetGenerateImage(cv2_scribble)
        shape_img = np.frombuffer(response.img_1, dtype=np.uint8).\
            reshape((self.img_size, self.img_size, 3))
        generated_image = np.frombuffer(response.img_2, dtype=np.uint8).\
            reshape((392, 262, 3))

        self.drawWidget.setShadowImage(shape_img)
        self.visWidget.update_vis_cv2(generated_image)

    def reset(self):
        self.drawWidget.reset()

    def move_stroke(self):
        self.drawWidget.move_stroke()

    def warp_stroke(self):
        self.drawWidget.warp_stroke()

    def draw_stroke(self):
        self.drawWidget.draw_stroke()

    def randomize(self):
        call_RandomizeNoise()
        self.generate()

    def scribble(self):
        self.drawWidget.scribble()

    def erase(self):
        self.drawWidget.erase()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GANGATEGui()
    window.setWindowTitle('iSketchNFill')
    window.setWindowFlags(window.windowFlags() & ~
                          Qt.WindowMaximizeButtonHint)   # fix window siz
    window.show()
    sys.exit(app.exec_())
