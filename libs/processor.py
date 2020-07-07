import math
import PIL.ImageQt
import cv2
import numpy as np
import scipy.misc as sm
from PyQt5 import QtCore
from libs.utils import *
from PyQt5.QtGui import QPixmap


class Processor:

    def __init__(self, step, alpha, gamma, detectors,context,flt,show_steps):
        self.context = context
        self.D_num = detectors
        self.scan_angle = 181
        self.alpha = alpha
        self.gamma = gamma
        self.step = step
        self.Em = np.zeros(shape=(int(self.scan_angle // step), detectors, 2), dtype='i8')
        self.Dt = np.zeros(shape=(int(self.scan_angle // step), detectors, 2), dtype='i8')
        self.org_shape = (0, 0)
        self.init_shape=(0,0)
        self.init_image=[]
        self.show_steps=show_steps
        x = np.arange(-20, 20)
        self.filter_kernel = list(map(fil, x))
        self.use_filter= flt
        self.sinograms = []
        self.outs = []
        self.rmse=[]

    def init_em(self, R):
        global global_test
        detectors = self.D_num
        for a in np.linspace(0, self.scan_angle, int(self.scan_angle // self.step), endpoint=False):
            j = int(a // self.step)
            temp = global_test.copy()
            angle = self.alpha + a
            E = (int(R * np.cos(math.radians(angle)) + R), int(R * np.sin(math.radians(angle)) + R))

            if detectors == 1:
                gamma = 0
                x = int(R * np.cos(math.radians(angle - (self.gamma / 2)) + np.pi) + R)
                y = int(R * np.sin(math.radians(angle - (self.gamma / 2)) + np.pi) + R)
                self.Em[j][0][0] = E[0]
                self.Em[j][0][1] = E[1]
                self.Dt[j][0] = [x, y]
            else:

                for i in range(detectors):
                    if i == 0:
                        x = int(R * np.cos(math.radians(angle - (self.gamma / 2)) + np.pi) + R)
                        y = int(R * np.sin(math.radians(angle - (self.gamma / 2)) + np.pi) + R)
                        xe = int(R * np.cos(math.radians(angle - (self.gamma / 2) + 180) + np.pi) + R)
                        ye = int(R * np.sin(math.radians(angle - (self.gamma / 2) + 180) + np.pi) + R)
                    else:

                        x = int(
                            R * np.cos(math.radians(
                                angle + 0 - (self.gamma / 2) + (i * self.gamma / (detectors - 1))) + np.pi) + R)
                        y = int(
                            R * np.sin(math.radians(
                                angle + 0 - (self.gamma / 2) + (i * self.gamma / (detectors - 1))) + np.pi) + R)
                        xe = int(
                            R * np.cos(math.radians(
                                angle + 180 - (self.gamma / 2) + (i * self.gamma / (detectors - 1))) + np.pi) + R)
                        ye = int(
                            R * np.sin(math.radians(
                                angle + 180 - (self.gamma / 2) + (i * self.gamma / (detectors - 1))) + np.pi) + R)

                    self.Dt[j][detectors - i - 1] = [x, y]
                    self.Em[j][i] = [xe, ye]
            for i in range(detectors):
                temp = cv2.drawMarker(temp, (self.Dt[j][i][0], self.Dt[j][i][1]), (255, 0, 0), 5)
                temp = cv2.drawMarker(temp, (self.Em[j][i][0], self.Em[j][i][1]), (255, 0, 0), 5)

    def get_sins_len(self):
        return len(self.sinograms)

    def get_imgs_len(self):
        return len(self.imgs)

    def get_sin(self,idx):
        return self.sinograms[idx]

    def get_img(self,idx):
        return self.imgs[idx]

    def get_rmse(self,idx):
        return self.rmse[idx]

    def prepare_image(self, img, scale):
        global global_test

        img = cv2.imread(img, 0)

        img = cv2.resize(img, (0, 0), fy=scale, fx=scale)
        self.init_shape=img.shape
        self.init_image = img.copy()
        dimensions = np.shape(img)
        size = max(dimensions)
        R = math.sqrt(2) * (size / 2)

        ver = int(abs(dimensions[1] - 2 * R) / 2) + 1
        hor = int(abs(dimensions[0] - 2 * R) / 2) + 1
        square_img = cv2.copyMakeBorder(img, hor, hor, ver, ver, cv2.BORDER_CONSTANT)
        global_test = square_img.copy()
        self.init_em(R)
        self.org_shape = np.shape(square_img)

        return square_img

    def radon_transform(self, square_img):
        step_size = int(np.ceil(self.scan_angle / self.step)) - 1
        iteration_count = int(np.ceil(self.scan_angle / self.step / 2))
        T = np.zeros((step_size, self.D_num))
        self.sinograms = np.zeros((iteration_count, step_size, self.D_num))
        idx_sinograms = 0
        for a in np.linspace(0, self.scan_angle, step_size, endpoint=False):
            i = int(a / self.step)

            res = np.zeros(self.D_num)

            for j in range(self.D_num):
                res[j] = scan_line(self.Em[i][j][0] , self.Em[i][j][1], self.Dt[i][j][0], self.Dt[i][j][1], square_img)
            T[i] = res
            Test = T.copy()
            Test = normalize(Test)

            if i % 2 == 0:
                self.sinograms[idx_sinograms] = Test.copy()
                idx_sinograms += 1
            if self.show_steps:
                Test = T.copy()
                Test = normalize(Test)
                q = QPixmap.fromImage(PIL.ImageQt.ImageQt(sm.toimage(Test)))
                w = self.context.sinogram_image.width()
                h = self.context.sinogram_image.height()
                self.context.sinogram_image.setPixmap(q.scaled(w,h,QtCore.Qt.KeepAspectRatio))
            else:
                if int(a / self.scan_angle * 100) <= 100:
                    self.context.sin_prog.setValue(int(a/self.scan_angle*100))

        if idx_sinograms != iteration_count:
            self.sinograms[idx_sinograms] = Test.copy()

        Test = T.copy()
        Test = normalize(Test)
        q = QPixmap.fromImage(PIL.ImageQt.ImageQt(sm.toimage(Test)))
        w = self.context.sinogram_image.width()
        h = self.context.sinogram_image.height()
        self.context.sinogram_image.setPixmap(q.scaled(w, h, QtCore.Qt.KeepAspectRatio))


        return T

    def iradon_transform(self, sinogram, mode=0, filtered=True):
        begin_x = int((self.org_shape[0]-self.init_shape[0])//2)
        begin_y = int((self.org_shape[1]-self.init_shape[1])//2)
        end_x = int(begin_x + self.init_shape[0])
        end_y = int(begin_y + self.init_shape[1])
        if self.use_filter:
            for i in range(len(sinogram)):
                sinogram[i] = custom_convolution(sinogram[i], self.filter_kernel)

        if mode == 1:
            out = np.zeros(self.org_shape)
            self.imgs = np.zeros((int(np.ceil((self.scan_angle / self.step)/2)),self.init_shape[0], self.init_shape[1]))
            self.rmse = np.zeros((int(np.ceil((self.scan_angle / self.step)/2))))
            idx_imgs = 0
            for a in np.linspace(0, self.scan_angle, int(np.ceil(self.scan_angle / self.step)) - 1, endpoint=False):
                i = int(a // self.step)
                for j in range(self.D_num):
                    scan_line(self.Em[i][j][0], self.Em[i][j][1], self.Dt[i][j][0], self.Dt[i][j][1], out, "back",
                              sinogram[i][j])
                Test1 = out.copy()
                Test =Test1[begin_x:end_x, begin_y:end_y]
                Test = normalize_255(Test).astype(np.uint8)
                Test = cv2.blur(Test, (7, 7))
                Test = normalize_255(Test).astype(np.uint8)

                pixel_sum=0

                if self.show_steps:
                    #
                    if i % 2 == 0:
                        self.imgs[idx_imgs] = Test.copy()
                        idx_imgs += 1

                    q = QPixmap.fromImage(PIL.ImageQt.ImageQt(sm.toimage(Test)))
                    w = self.context.out_image.width()
                    h = self.context.out_image.height()
                    self.context.out_image.setPixmap(q.scaled(w, h, QtCore.Qt.KeepAspectRatio))
                else:
                    if(int(a / self.scan_angle * 100)<=100):
                        self.context.out_prog.setValue(int(a / self.scan_angle * 100))

            out = out[begin_x:end_x, begin_y:end_y]
            out = normalize_255(out).astype(np.uint8)
            out = cv2.blur(out, (7, 7))
            out = normalize_255(out).astype(np.uint8)

            self.imgs[len(self.sinograms) - 1] = out

            q = QPixmap.fromImage(PIL.ImageQt.ImageQt(sm.toimage(out)))
            w = self.context.out_image.width()
            h = self.context.out_image.height()
            self.context.out_image.setPixmap(q.scaled(w, h, QtCore.Qt.KeepAspectRatio))

            pixel_sum=0
            for n in range(out.shape[0]):
                for m in range(out.shape[1]):
                    pixel_sum += math.pow(int(out[n][m]) - int(self.init_image[n][m]), 2)
            pixel_sum /= out.shape[0] * out.shape[1]
            pixel_sum = math.sqrt(pixel_sum)
            self.rmse[len(self.rmse)-1] = pixel_sum
            self.context.rmse.setText(str(pixel_sum))

        return out