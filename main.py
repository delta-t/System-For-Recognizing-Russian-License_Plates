"""
Using Haar cascade classifier for detection russian license plates
"""
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from mygui import Ui_MyApp
from imutils.perspective import four_point_transform
from PIL import Image
from os import path

import sys
import cv2
import numpy as np
import imutils
import scipy.fftpack
import pytesseract
import re


class VideoThread(QThread):
    image = pyqtSignal(QImage)

    def __init__(self, haar_cascade_filepath, num, flag, address=0):
        super(VideoThread, self).__init__()
        self.cap = cv2.VideoCapture(address)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.flag = flag
        self.counter = 0
        self.license_number = None
        self.stop_it = False
        self._red = (0, 0, 255)
        self.width = 2
        self._min_size = (10, 30)
        self.show_num = num

    def run(self):
        while self.cap.isOpened():
            if self.stop_it:
                return

            # Reading and preprocessing
            _, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray_filtered = cv2.medianBlur(gray, 5)
            h, w = gray_filtered.shape
            gray_normalized = np.zeros((h, w))
            gray_normalized = cv2.normalize(gray_filtered, gray_normalized, 0, 255, cv2.NORM_MINMAX)
            
            # Looking for a license plate
            plaques = []
            plaques = self.classifier.detectMultiScale(gray_normalized, scaleFactor=1.3, minNeighbors=4)
            if len(plaques) > 0:
                for (x, y, w, h) in plaques:
                    # Crop the frame
                    roi_color = frame[y:y + h, x:x + w]
                    gray_roi = gray[y:y + h, x:x + w]

                    # Preprocessing of ROI for detect plate border
                    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
                    edged = cv2.Canny(blurred, 75, 200)
                    cntrs = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cntrs = imutils.grab_contours(cntrs)
                    numCnt = None

                    # if found any rectangle, consider it plate
                    if len(cntrs) > 0:
                        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)
                        for c in cntrs:
                            if cv2.isContourConvex(c):
                                peri = cv2.arcLength(c, True)
                                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                                if len(approx) == 4:
                                    numCnt = approx
                                    break
                    
                    # trying to transform the plate for recognition
                    if numCnt is not None:
                        # Crop the roi
                        license_plate = four_point_transform(gray_roi, numCnt.reshape(4, 2))
                        license_plate = imutils.resize(license_plate, width=300)

                        # Thresholding license plate
                        final_image = cv2.adaptiveThreshold(license_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, 115, 1)

                        # Using Pytesseract for recognition, draw license plate's contour and print the number
                        check1_text = pytesseract.image_to_string(final_image, config='--psm 8')

                        # Filtration license symbols
                        text1 = check1_text.upper()
                        text1 = re.sub(r'[^\w\s]', '', text1)

                        if 10 > len(text1) > 5:
                            self.flag = True
                            self.show_num.setText("Номер авто: " + text1)
                            self.show_num.adjustSize()

                        # Clear it
                        text1 = None

                    if not self.flag:
                        # Number of rows and columns
                        rows = gray_roi.shape[0]
                        cols = gray_roi.shape[1]

                        # Convert image to 0 to 1, then do log(1 + I)
                        gray_log = np.log1p(np.array(gray_roi, dtype="float") / 255)

                        # Create Gaussian mask of sigma = 10
                        M = 2 * rows + 1
                        N = 2 * cols + 1
                        sigma = 10

                        (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
                        center_x = np.ceil(N / 2)
                        center_y = np.ceil(M / 2)
                        gaussian_numerator = (X - center_x) ** 2 + (Y - center_y) ** 2

                        # low pass and high pass filters
                        h_low = np.exp(-gaussian_numerator / (2 * sigma * sigma))
                        h_high = 1 - h_low

                        # Move origin of filters so that it's at the top left corner to
                        # match with the input image
                        h_low_shift = scipy.fftpack.ifftshift(h_low.copy())
                        h_high_shift = scipy.fftpack.ifftshift(h_high.copy())

                        # Filter the image and crop
                        img_filt = scipy.fftpack.fft2(gray_log.copy(), (M, N))
                        img_out_low = scipy.real(scipy.fftpack.ifft2(img_filt.copy() * h_low_shift, (M, N)))
                        img_out_high = scipy.real(scipy.fftpack.ifft2(img_filt.copy() * h_high_shift, (M, N)))
                        # Set scaling factors and add
                        gamma1 = 0.3
                        gamma2 = 1.5
                        img_out = gamma1 * img_out_low[0:rows, 0:cols] + gamma2 * img_out_high[0:rows, 0:cols]

                        # Anti-log then rescale to [0, 1]
                        img_hmf = np.expm1(img_out)
                        img_hmf = (img_hmf - np.min(img_hmf)) / (np.max(img_hmf) - np.min(img_hmf))
                        img_hmf2 = np.array(255 * img_hmf, dtype="uint8")

                        # Thresholding the image - Anything below intensity 65 gets set to white
                        img_thresh = img_hmf2 < 80
                        img_thresh = 255 * img_thresh.astype("uint8")

                        # Clear off the border. Choose a border radius of 5 pixels
                        img_clear = self.im_clear_border(img_thresh, 5)

                        # Eliminate regions that have areas below 120 pixels
                        img_open = self.bw_area_open(img_clear, 5)
                        img_open = cv2.bitwise_not(img_open)
                        check2_text = pytesseract.image_to_string(img_open, config='--psm 8')
                        text2 = check3_text.upper()
                        text2 = re.sub(r'[^\w\s]', '', text2)
                        if 10 > len(text2) > 5:
                            self.show_num.setText("License plate: " + text2)
                            self.show_num.adjustSize()
                        text2 = None
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self._red, self.width)
            height, width, _ = frame.shape
            bytesPerLine = 3 * width
            frame = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            # frame = frame.scaled(640, 480, Qt.KeepAspectRatio)
            self.image.emit(frame)

    @staticmethod
    def bw_area_open(imgBW, areaPixels):
        # given a black and white image, first find all of its contours
        imgBW_copy = imgBW.copy()
        contours, _ = cv2.findContours(imgBW_copy.copy(), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # for each contours, determine its local occupying area
        for idx in np.arange(len(contours)):
            area = cv2.contourArea(contours[idx])
            if 0 <= area <= areaPixels:
                cv2.drawContours(imgBW_copy, contours, idx, (0, 0, 0), -1)
        return imgBW_copy

    @staticmethod
    def im_clear_border(imgBW, radius):
        # given a black and white image, first find all of its contours
        imgBW_copy = imgBW.copy()
        contours, _ = cv2.findContours(imgBW_copy.copy(), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # get dimensions of image
        imgRows = imgBW.shape[0]
        imgCols = imgBW.shape[1]
        contourList = []

        # for each contourArea
        for idx in np.arange(len(contours)):
            # get i'th contour
            cnt = contours[idx]

            # look at each point in the contour
            for pt in cnt:
                rowCnt = pt[0][1]
                colCnt = pt[0][0]
                # if this is within the radius of the border
                # this contour goes bye-bye
                check1 = (0 <= rowCnt < radius) or (imgRows - 1 - radius <= rowCnt < imgRows)
                check2 = (0 <= colCnt < radius) or (imgCols - 1 - radius <= colCnt < imgCols)
                if check1 or check2:
                    contourList.append(idx)
                    break
        for idx in contourList:
            cv2.drawContours(imgBW_copy, contours, idx, (0, 0, 0), -1)
        return imgBW_copy


class MyWindow(QMainWindow):
    def __init__(self, haar_cascade_filepath, parent=None):
        super(MyWindow, self).__init__(parent)
        self.ui = Ui_MyApp()
        self.ui.setupUi(self)
        self.ui.number_plate.setText("License plate: - ")
        self.ui.number_plate.adjustSize()
        self.ui.start_app_button.clicked.connect(self.start_app)
        self.ui.stop_app_button.clicked.connect(self.stop_app)
        self.start_flag = False

        self.video = VideoThread(address=0, haar_cascade_filepath=haar_cascade_filepath,
                                 num=self.ui.number_plate, flag=self.start_flag)
        self.video.image.connect(self.setFrame)

    def setFrame(self, frame):
        pixmap = QPixmap.fromImage(frame)
        self.ui.license_detect_frame.setPixmap(pixmap)

    def start_app(self):
        if not self.start_flag:
            self.start_flag = True
            self.video.stop_it = False
            self.video.start()

    def stop_app(self):
        self.video.stop_it = True
        self.start_flag = False

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.video.stop_it = True
            self.video.cap.release()
            event.accept()
        else:
            event.ignore()


def main(haar_cascade_filepath):
    app = QApplication(sys.argv)
    application = MyWindow(haar_cascade_filepath=haar_cascade_filepath)
    application.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = path.join(script_dir, 'haarcascade_russian_plate_number.xml')
    cascade_filepath = path.abspath(cascade_filepath)
    main(haar_cascade_filepath=cascade_filepath)
