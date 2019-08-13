"""
Данный скрипт использует обученные нейронные сети детекции и идентификации российских автомобильных номеров 
"""
# import necessary packages
import datetime
import os
import socket
import sys
import threading
import time
import warnings
from datetime import datetime as dt

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow
from imutils import resize

from BoxPanel import Ui_MainPanel

warnings.filterwarnings('ignore')
close_flag = False


# send a message to command module
def send_message(cmd, sock):
    try:
        sock.send(cmd)
        print("Output command:", cmd)
        return True
    except OSError:
        print("Error, module is not responding")
        return False


# compare a license number with white numbers
def check_whitelist(license_num):
    # initially, the door is closed
    hold_the_door = True
    # read the whitelist
    white_number = set(line.strip() for line in open('number_plate_base.txt', 'r'))
    # compare license with each white number
    for number in white_number:
        k = 0
        for idx in range(min(len(license_num), len(number))):
            # get the "value" of the character
            ordinal_license, ordinal_number = str(ord(license_num[idx].upper())), \
                                                  str(ord(number[idx].upper()))
            # if the "value" are identical, check the next pairs
            if (ordinal_license == "O" and ordinal_number == "0") \
                    or (ordinal_license == "0" and ordinal_number == "O") \
                    or ordinal_license == ordinal_number:
                k += 1

            # if sum of k is bigger than x characters - open the door
            if k > 6:
                hold_the_door = False
                return hold_the_door, number
    # keep the barrier closed
    return hold_the_door, None


# Video Capture with asynchronous method
class VideoCaptureAsync:
    def __init__(self, src, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self):
        self.stop()
        self.cap.release()


# Application Board
class MyWindow(QMainWindow):
    def __init__(self, sock):
        super(MyWindow, self).__init__()
        self.ui = Ui_MainPanel()
        self.ui.setupUi(self)
        self.ui.barrierStatus.setText("Шлагбаум: - ")
        self.ui.barrierStatus.adjustSize()
        self.ui.carNumber.setText('Номер авто: - ')
        self.ui.carNumber.adjustSize()
        self.ui.outCar.setText('Авто на выезде: - ')
        self.ui.outCar.adjustSize()
        self.ui.openButton.clicked.connect(self.open_barrier)
        self.ui.closeButton.clicked.connect(self.close_barrier)
        self.port = sock
        self.OPEN = b'2'
        self.CLOSE = b'1'

    def open_barrier(self):
        if send_message(self.OPEN, sock=self.port):
            self.ui.barrierStatus.setText('Шлагбаум: открыт')
            self.ui.barrierStatus.adjustSize()
        else:
            self.ui.barrierStatus.setText('Команда \nне принята')
            self.ui.barrierStatus.adjustSize()

    def close_barrier(self):
        if send_message(self.CLOSE, sock=self.port):
            self.ui.barrierStatus.setText('Шлагбаум: закрыт')
            self.ui.barrierStatus.adjustSize()
        else:
            self.ui.barrierStatus.setText('Команда \nне принята')
            self.ui.barrierStatus.adjustSize()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Вы точно хотите закрыть приложение?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            global close_flag
            close_flag = True
            event.accept()
        else:
            event.ignore()


def main(ss):
    trigger = datetime.time(hour=10, minute=0, second=0, microsecond=0)

    # specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
    NOMEROFF_NET_DIR = os.path.abspath('../')
    MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
    MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')
    MASK_RCNN_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/mask_rcnn_numberplate_0700.h5")
    OPTIONS_MODEL_PATH = os.path.join(NOMEROFF_NET_DIR, "models/numberplate_options_2019_03_05.h5")

    # If you use gpu version tensorflow please change model to gpu version named like *-gpu.pb
    mode = "gpu"
    OCR_NP_RU_TEXT = os.path.join(NOMEROFF_NET_DIR, "models/anpr_ocr_ru_3-{}.h5".format(mode))

    sys.path.append(NOMEROFF_NET_DIR)

    # Import license plate recognition tools.
    from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing

    # Initialize npdetector with default configuration file.
    nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)

    # Load weights in keras format.
    nnet.loadModel(MASK_RCNN_MODEL_PATH)

    # Initialize rect detector with default configuration file.
    rectDetector = RectDetector()

    # Initialize text detector.
    # Also you may use gpu version models.
    textDetector = TextDetector({
        "ru": {
            "for_regions": ["ru"],
            "model_path": OCR_NP_RU_TEXT
        }
    })

    # Initialize train detector.
    optionsDetector = OptionsDetector()
    optionsDetector.load(OPTIONS_MODEL_PATH)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    main_directory = os.path.join(script_dir, "data")
    
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    
    car_snapshot = len(os.listdir(main_directory))
    print(car_snapshot)
    
    caffe_net_filepath = os.path.join(script_dir, 'MobileNetSSD_deploy.caffemodel')
    caffe_net_filepath = os.path.abspath(caffe_net_filepath)
    proto_filepath = os.path.join(script_dir, 'MobileNetSSD_deploy.prototxt.txt')
    proto_filepath = os.path.abspath(proto_filepath)
    car_net = cv2.dnn.readNetFromCaffe(proto_filepath, caffe_net_filepath)
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    limit_for_confidence = 0.7
    number_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    src0 = 'rtsp://...'  # outside camera
    src1 = 'rtsp://...'  # inside camera
    inc_capture = VideoCaptureAsync(src0).start()
    out_capture = VideoCaptureAsync(src1).start()
    time.sleep(3.0)
    application = MyWindow(sock=ss)
    application.show()
    flag_cnt = False
    flag_outside = False
    start_save = True
    clear_log_files = dt.month  # clear logging every month
    cnt = 0
    file = open(script_dir + "/found_numbers.txt", 'a')
    while True:
        # clear history every month
        if dt.month != clear_log_files:
            clear_log_files = dt.month
            with open(script_dir + "/found_numbers.txt", 'w'):
                pass
            for the_file in os.listdir(main_directory):
                file_path = os.path.join(main_directory, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

        # getting the frame from incoming camera and frame preprocessing
        try:
            _, frame = inc_capture.read()
        except AttributeError:
            print("Incoming camera isn't responding")
            break
        frame = resize(frame, width=720)
        # Crop the ROI
        height_frame, width_frame = frame.shape[:2]
        frame = frame[int(0.3 * height_frame):height_frame, 0:width_frame]
        
        # if we get the frame in grayscale format (night mode enable) - convert it to bgr
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except cv2.error:
            pass

        # getting the frame from outcoming camera and frame preprocessing
        try:
            _, shot = out_capture.read()
        except AttributeError:
            print("Outcoming camera isn't responding")
            break
        shot = resize(shot, width=720)
        # Crop the ROI
        height_shot, width_shot = shot.shape[:2]
        shot = shot[int(0.3 * height_shot):height_shot, 0:width_shot]
        
        # if we get the frame in grayscale format (night mode enable) - convert it to bgr
        try:
            shot = cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR)
        except cv2.error:
            pass

        copy_frame = frame.copy()
        copy_shot = shot.copy()

        # Pass the blob through the network and obtain the detections and predictions
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        car_net.setInput(blob)
        detections = car_net.forward()

        (H, W) = shot.shape[:2]
        blobbed = cv2.dnn.blobFromImage(cv2.resize(shot, (300, 300)),
                                        0.007843, (300, 300), 127.5)
        car_net.setInput(blobbed)
        discoveries = car_net.forward()
        number_plate = ''
        # analyze incoming camera frame
        for ind in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the predictions
            confidence = detections[0, 0, ind, 2]
            # Filter out weak detections by ensuring the confidence is greater than
            # the minimum confidence
            if confidence > limit_for_confidence:
                # Extract the index of the class labels from detections, then compute
                # the (x, y)-coordinates of the bounding box for the object
                idx = int(detections[0, 0, ind, 1])
                box = detections[0, 0, ind, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding boxes for everything car
                if CLASSES[idx] == 'car':
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    car_frame = frame[startY:endY, startX:endX]
                    car_number = car_frame.copy()

                    # looking for a license plate on car frame
                    plaques = []
                    plaques = number_cascade.detectMultiScale(car_number, scaleFactor=1.3,
                                                             minNeighbors=4)
                    # if any license plate has been found - try to recognize its
                    if len(plaques) > 0 and flag_cnt is False:
                        start = time.time()
                        try:
                            car_number = resize(car_number, width=1080)
                        except cv2.error:
                            print('cv2 error')
                        except ZeroDivisionError:
                            print('zero division')

                        try:
                            NP = nnet.detect([car_number])
                            # Generate image mask.
                            cv_img_masks = filters.cv_img_mask(NP)
                            # Detect points.
                            arrPoints = rectDetector.detect(cv_img_masks)
                            zones = rectDetector.get_cv_zonesBGR(car_number, arrPoints)

                            # find standart
                            # Added a classifier (isHiddenIds) for determining the fact of hide text of number,
                            # in order not to recognize a deliberately damaged license plate image.
                            regionIds, isHiddenIds = optionsDetector.predict(zones)
                            regionNames = optionsDetector.getRegionLabels(regionIds)

                            # find text with postprocessing by standart
                            textArr = textDetector.predict(zones, regionNames)
                            textArr = textPostprocessing(textArr, regionNames)
                            number_plate = ''.join(textArr)
                            for (xx, yy, ww, hh) in plaques:
                                cv2.rectangle(copy_frame, (startX + xx, startY + yy),
                                              (startX + xx + ww, startY + yy + hh), (0, 0, 255), 2)

                            # if license plate has been recognized - compare the number with white numbers
                            if len(number_plate) > 0:
                                print(number_plate)
                                # if not flag_cnt:
                                flag, show = check_whitelist(number_plate)
                                if not flag:
                                    sending_flag = send_message(b'2', ss)
                                    if show is not None:
                                        application.ui.carNumber.setText('Номер:' + number_plate)
                                        application.ui.carNumber.adjustSize()
                                    if sending_flag:
                                        application.ui.barrierStatus.setText('Шлагбаум: открыт')
                                        application.ui.barrierStatus.adjustSize()
                                        flag_cnt = True
                                    else:
                                        application.ui.barrierStatus.setText('Команда \nне принята')
                                        application.ui.barrierStatus.adjustSize()
                        except ZeroDivisionError:
                            print('zero')
                        except RecursionError:
                            print('Recursion Error')
                        finish = time.time()
                        print(finish - start, 'at:', datetime.datetime.now(), sep=' ')
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(copy_frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, COLORS[idx], 2)

        # analyze outcoming camera frame
        for index in np.arange(0, discoveries.shape[2]):
            # extract the confidence (i.e., probability) associated with the predictions
            confidence = discoveries[0, 0, index, 2]
            # Filter out weak detections by ensuring the confidence is greater than
            # the minimum confidence
            if confidence > limit_for_confidence:
                # Extract the index of the class labels from detections, then compute
                # the (x, y)-coordinates of the bounding box for the object
                idx = int(discoveries[0, 0, index, 1])
                box = discoveries[0, 0, index, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding boxes for everything car
                if CLASSES[idx] == 'car':
                    # heightShot, widthShot = shot.shape[:2]
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    car_shot = shot[startY:endY, startX:endX]
                    car_area = car_shot.copy()
                    height_car, width_car = car_area.shape[:2]

                    # cv2.rectangle(frame, (startX, startY), (endX, endY),
                    #              COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(copy_shot, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, COLORS[idx], 2)

                    if not flag_cnt:
                        if height_car > 0.3 * H or width_car > 0.3 * W:
                            if not flag_outside:
                                sending_flag = send_message(b'2', ss)
                                if sending_flag:
                                    application.ui.outCar.setText("Авто на выезде: да")
                                    application.ui.outCar.adjustSize()
                                    application.ui.barrierStatus.setText('Шлагбаум: открыт')
                                    application.ui.barrierStatus.adjustSize()
                                    flag_outside = True
                                else:
                                    application.ui.barrierStatus.setText('Команда \nне принята')
                                    application.ui.barrierStatus.adjustSize()
                                flag_cnt = True

        if start_save:
            if len(number_plate) > 0:
                number_with_time = str(number_plate) + " at: " + str(dt.now())
                file.write("%s\n" % str(number_with_time))
                car_snapshot += 1
                cv2.imwrite(main_directory + "/" + str(car_snapshot) + ".jpg", frame)
                # time.sleep(5.0)
                print("The frame has been saved")
        cv2.imshow('Incoming', copy_frame)
        cv2.imshow('Outcoming', copy_shot)
        time.sleep(0.2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if close_flag:
            break

        # while flag_cnt is True:
        if flag_cnt:
            cnt += 1
            # time.sleep(1.0)
            if cnt > 100:
                sending_flag = send_message(b'1', ss)
                if sending_flag:
                    application.ui.barrierStatus.setText('Шлагбаум: закрыт')
                    application.ui.outCar.setText("Авто на выезде: нет")
                    application.ui.outCar.adjustSize()
                    application.ui.barrierStatus.adjustSize()
                    flag_outside = False
                    flag_cnt = False
                else:
                    application.ui.barrierStatus.setText('Команда \nне принята')
                    application.ui.barrierStatus.adjustSize()
                cnt = 0

    # release all cameras
    inc_capture.__exit__()
    out_capture.__exit__()
    file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    app = QApplication(sys.argv)
    try:
        TCP_IP = 'XXX.XXX.XXX.XXX'
        TCP_PORT = 5005
        BUFFER_SIZE = 1
        s.connect((TCP_IP, TCP_PORT))
        main(ss=s)
    except TimeoutError:
        for i in range(5):
            print("Попытка соединения с модулем управления шлагбаумом была безуспешной, приложение закроется через:",
                  str(5 - i))
            time.sleep(1.0)

    finally:
        s.close()
