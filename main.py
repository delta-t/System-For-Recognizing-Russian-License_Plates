"""
Using Haar cascade classifier for detection russian license plates
"""
import sys
from os import path

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow

from CarDetection import VideoThread
from mygui import UiMyApp


class MyWindow(QMainWindow):
    def __init__(self, haar_cascade_filepath, parent=None):
        super(MyWindow, self).__init__(parent)
        self.ui = UiMyApp()
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
