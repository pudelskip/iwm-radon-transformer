import math
import os
import sys

import PIL.ImageQt
import scipy.misc as sm
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QProgressBar
from PyQt5.uic import loadUi
from libs.processor import Processor


class MyThread(QThread):

    def __init__(self,img,scale,context):
        QThread.__init__(self)
        self.img=img
        self.scale=scale
        self.context = context

    def __del__(self):
        self.wait()

    def run(self):

        prepared = self.context.image_processor.prepare_image(self.img, self.scale)
        height, width = prepared.shape

        self.context.label.setText("working...")
        radon = self.context.image_processor.radon_transform(prepared)
        self.context.sin_prog.setVisible(False)
        self.context.slider.setMaximum(self.context.image_processor.get_sins_len() - 1)
        self.context.slider.setMinimum(0)
        if(self.context.show_process_steps):
            self.context.slider.setVisible(True)
            self.context.label_sin.setVisible(True)


        out = self.context.image_processor.iradon_transform(radon, mode=1, filtered=True)
        self.context.out_prog.setVisible(False)
        self.context.sliderR.setMinimum(0)
        self.context.sliderR.setMaximum(self.context.image_processor.get_imgs_len() - 1)
        if(self.context.show_process_steps):
            self.context.sliderR.setVisible(True)
            self.context.label_img.setVisible(True)

        self.context.rmse.setVisible(True)
        self.context.processing = False
        self.context.pushButton.setEnabled(True)
        self.context.label.setText("ready")


class MyUi(QDialog):
    def __init__(self):
        super(MyUi, self).__init__()
        loadUi("myui.ui",self)
        self.pushButton.clicked.connect(self.start)
        files = os.listdir("./img")
        self.processing = False
        self.comboBox.addItems(files)
        self.comboBox.currentIndexChanged.connect(self.show_in_img)
        self.ready=False
        self.show_process_steps = False
        self.current_image=""
        self.image_processor = None
        self.slider.valueChanged.connect(self.slider_changed)
        self.sliderR.valueChanged.connect(self.sliderR_changed)
        self.slider.setVisible(False)
        self.sliderR.setVisible(False)
        self.label_sin.setVisible(False)
        self.label_img.setVisible(False)
        self.rmse.setVisible(False)
        self.sin_prog.setVisible(False)
        self.out_prog.setVisible(False)
        self.label.setText("ready")
        a = QProgressBar()


    @pyqtSlot()
    def start(self):
        if(not self.processing):
            if(self.ready):
                if(self.lineEdit.text() != ""):
                    self.pushButton.setEnabled(False)
                    self.slider.setVisible(False)
                    self.sliderR.setVisible(False)
                    self.label_sin.setVisible(False)
                    self.label_img.setVisible(False)
                    self.rmse.setVisible(False)
                    self.sinogram_image.setText("Miejsce na sinogram")
                    self.out_image.setText("Miejsce na obraz wyjściowy")

                    filter = self.checkBox.isChecked()
                    step = float(self.lineEdit.text())
                    gamma = int(self.lineEdit_3.text())
                    det = int(self.lineEdit_2.text())
                    self.show_process_steps = self.steps.isChecked()

                    self.image_processor = Processor(step=step, alpha=0, gamma=gamma, detectors=det, context=self,flt=filter,show_steps=self.show_process_steps)
                    if(not self.show_process_steps):
                        self.sin_prog.setVisible(True)
                        self.out_prog.setVisible(True)

                    self.processing = True
                    self.get_thread = MyThread(self.current_image,1.0,self)
                    self.get_thread.start()
                else:
                    QMessageBox.warning(self, "Brak parametrów", "Proszę wpisać wszystkie parametry")
            else:
                QMessageBox.warning(self,"Brak obrazu","Proszę wybrać obraz")



    def show_in_img(self):
        self.current_image="./img/"+self.comboBox.currentText()
        pixmap = QPixmap("./img/"+self.comboBox.currentText())
        w = self.in_image.width()
        h = self.in_image.height()
        self.in_image.setPixmap(pixmap.scaled(w,h,QtCore.Qt.KeepAspectRatio))
        self.ready=True


    def slider_changed(self):
        i = self.slider.value()
        q = QPixmap.fromImage(PIL.ImageQt.ImageQt(sm.toimage(self.image_processor.get_sin(i))))
        w = self.sinogram_image.width()
        h = self.sinogram_image.height()
        self.sinogram_image.setPixmap(q.scaled(w, h, QtCore.Qt.KeepAspectRatio))
        # print(i)
    def sliderR_changed(self):
        i = self.sliderR.value()
        q = QPixmap.fromImage(PIL.ImageQt.ImageQt(sm.toimage(self.image_processor.get_img(i))))
        w = self.out_image.width()
        h = self.out_image.height()
        self.out_image.setPixmap(q.scaled(w, h, QtCore.Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MyUi()
    widget.setWindowTitle("Radon transform visualizer")
    widget.show()
    sys.exit(app.exec_())