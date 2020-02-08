import os
import sys
import cv2
import time
import copy
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
from ui_shadow_draw.ui_recorder import UIRecorder
from ui_shadow_draw.gangate_draw import GANGATEDraw
from ui_shadow_draw.gangate_vis import GANGATEVis


channel = grpc.insecure_channel('192.168.1.131:50066')
stub = image_transfer_pb2_grpc.ImageTransferStub(channel)


def get_Image(image, label, shadow):
    return image_transfer_pb2.Image(data=image, label=label, shadow=shadow)


def call_GetGenerateImage(image_np, label, shadow):
    response = stub.GetGenerateImage(get_Image(image_np.tobytes(), label, shadow))
    return response


def call_RandomizeNoise():
    stub.RandomizeNoise(image_transfer_pb2.Empty())


class GANGATEGui(QWidget):
    def __init__(self, win_size=256, img_size=256):
        QWidget.__init__(self)

        self.win_size = win_size
        self.img_size = img_size

        self.drawWidget = GANGATEDraw(
            win_size=self.win_size, img_size=self.img_size)
        self.drawWidget.setFixedSize(win_size, win_size)

        self.visWidget_color = GANGATEVis(
            win_size=self.win_size, img_size=self.img_size,
            disable_browser=True)
        self.visWidget_color.setFixedSize(win_size, win_size)

        vbox = QVBoxLayout()

        self.drawWidgetBox = QGroupBox()
        self.drawWidgetBox.setTitle('Drawing Pad')
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(self.drawWidget)
        self.drawWidgetBox.setLayout(vbox_t)
        vbox.addWidget(self.drawWidgetBox)

        self.labelId = 6

        self.bBasketball = QRadioButton("Basketball")
        self.bBasketball.setToolTip(
            "This button enables generation of a Basketball")

        self.bSoccer = QRadioButton("Soccer")
        self.bSoccer.setToolTip("This button enables generation of a Soccer")

        self.bWatermelon = QRadioButton("Watermelon")
        self.bWatermelon.setToolTip(
            "This button enables generation of a Watermelon")

        self.bOrange = QRadioButton("Orange")
        self.bOrange.setToolTip("This button enables generation of a Orange")

        self.bCookie = QRadioButton("Cookie")
        self.bCookie.setToolTip("This button enables generation of a Cookie")

        self.bMoon = QRadioButton("Moon")
        self.bMoon.setToolTip("This button enables generation of a Moon")

        self.bStrawberry = QRadioButton("Strawberry")
        self.bStrawberry.setToolTip(
            "This button enables generation of a Strawberry")

        self.bPineapple = QRadioButton("Pineapple")
        self.bPineapple.setToolTip(
            "This button enables generation of a Pineapple")

        self.bCupcake = QRadioButton("Cupcake")
        self.bCupcake.setToolTip("This button enables generation of a Cupcake")

        self.bChicken = QRadioButton("Fried Chicken")
        self.bChicken.setToolTip("This button enables generation of a Chicken")

        bhbox = QGridLayout()  # QHBoxLayout()
        bGroup = QButtonGroup(self)

        bGroup.addButton(self.bBasketball)
        bGroup.addButton(self.bSoccer)
        bGroup.addButton(self.bWatermelon)
        bGroup.addButton(self.bOrange)
        bGroup.addButton(self.bCookie)
        bGroup.addButton(self.bMoon)
        bGroup.addButton(self.bStrawberry)
        bGroup.addButton(self.bPineapple)
        bGroup.addButton(self.bCupcake)
        bGroup.addButton(self.bChicken)

        bhbox.addWidget(self.bBasketball, 0, 0)
        bhbox.addWidget(self.bSoccer, 1, 0)
        bhbox.addWidget(self.bWatermelon, 2, 0)
        bhbox.addWidget(self.bOrange, 3, 0)
        bhbox.addWidget(self.bCookie, 4, 0)
        bhbox.addWidget(self.bMoon, 0, 1)
        bhbox.addWidget(self.bStrawberry, 1, 1)
        bhbox.addWidget(self.bPineapple, 2, 1)
        bhbox.addWidget(self.bCupcake, 3, 1)
        bhbox.addWidget(self.bChicken, 4, 1)

        self.bGenerate = QPushButton('Generate !')
        self.bGenerate.setToolTip(
            "This button generates the final image to render")

        self.bReset = QPushButton('Reset !')
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

        self.bEnableShadows = QCheckBox('Enable Shadows')
        self.bEnableShadows.toggle()

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)

        vbox4 = QVBoxLayout()
        self.visWidgetBox = QGroupBox()
        self.visWidgetBox.setTitle('Generations')

        vbox_t4 = QVBoxLayout()
        vbox_t4.addWidget(self.visWidget_color)
        self.visWidgetBox.setLayout(vbox_t4)
        vbox4.addWidget(self.visWidgetBox)

        hbox.addLayout(vbox4)
        hbox.addLayout(bhbox)

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
        bhbox_controls.addWidget(self.bEnableShadows, 0, 5)

        hbox.addLayout(bhbox)

        controlBox = QGroupBox()
        controlBox.setTitle('Controls')

        controlBox.setLayout(bhbox_controls)

        vbox_final = QVBoxLayout()
        vbox_final.addLayout(hbox)
        vbox_final.addWidget(controlBox)
        self.setLayout(vbox_final)

        self.bPineapple.setChecked(True)

        self.bDrawStroke.setChecked(True)

        self.enable_shadow = True

        self.bBasketball.clicked.connect(self.Basketball)
        self.bSoccer.clicked.connect(self.Soccer)
        self.bWatermelon.clicked.connect(self.Watermelon)
        self.bOrange.clicked.connect(self.Orange)
        self.bCookie.clicked.connect(self.Cookie)
        self.bMoon.clicked.connect(self.Moon)
        self.bStrawberry.clicked.connect(self.Strawberry)
        self.bPineapple.clicked.connect(self.Pineapple)
        self.bCupcake.clicked.connect(self.Cupcake)
        self.bChicken.clicked.connect(self.Chicken)
        self.bGenerate.clicked.connect(self.generate)
        self.bReset.clicked.connect(self.reset)
        self.bRandomize.clicked.connect(self.randomize)
        self.bMoveStroke.clicked.connect(self.move_stroke)
        self.bWarpStroke.clicked.connect(self.warp_stroke)
        self.bDrawStroke.clicked.connect(self.draw_stroke)
        self.bEnableShadows.stateChanged.connect(self.toggle_shadow)

    def toggle_shadow(self, state):
        if state == Qt.Checked:
            self.enable_shadow = True
        else:
            self.enable_shadow = False
        self.generate()

    def Basketball(self):
        self.labelId = 0

    def Soccer(self):
        self.labelId = 7

    def Watermelon(self):
        self.labelId = 9

    def Orange(self):
        self.labelId = 5

    def Cookie(self):
        self.labelId = 2

    def Moon(self):
        self.labelId = 4

    def Strawberry(self):
        self.labelId = 8

    def Pineapple(self):
        self.labelId = 6

    def Cupcake(self):
        self.labelId = 3

    def Chicken(self):
        self.labelId = 1

    def generate(self):
        cv2_scribble = self.drawWidget.getDrawImage()  # 256x256x3
        response = call_GetGenerateImage(
            cv2_scribble, self.labelId, int(self.enable_shadow))
        shape_img = np.frombuffer(response.img_1, dtype=np.uint8).\
            reshape((self.img_size, self.img_size, 3))
        generated_image = np.frombuffer(response.img_2, dtype=np.uint8).\
            reshape((self.img_size, self.img_size, 3))

        self.drawWidget.setShadowImage(shape_img)
        self.visWidget_color.update_vis_cv2(generated_image)

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
