# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class UiMyApp(object):
    def setupUi(self, MyApp):
        MyApp.setObjectName("MyApp")
        MyApp.resize(660, 600)
        self.start_app_button = QtWidgets.QPushButton(MyApp)
        self.start_app_button.setGeometry(QtCore.QRect(60, 510, 90, 80))
        self.start_app_button.setObjectName("start_app_button")
        self.stop_app_button = QtWidgets.QPushButton(MyApp)
        self.stop_app_button.setGeometry(QtCore.QRect(510, 510, 90, 80))
        self.stop_app_button.setObjectName("stop_app_button")
        self.number_plate = QtWidgets.QLabel(MyApp)
        self.number_plate.setGeometry(QtCore.QRect(270, 520, 120, 30))
        self.number_plate.setFrameShape(QtWidgets.QFrame.Box)
        self.number_plate.setText("")
        self.number_plate.setObjectName("number_plate")
        self.license_detect_frame = QtWidgets.QLabel(MyApp)
        self.license_detect_frame.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.license_detect_frame.setFrameShape(QtWidgets.QFrame.Box)
        self.license_detect_frame.setText("")
        self.license_detect_frame.setObjectName("license_detect_frame")

        self.retranslateUi(MyApp)
        QtCore.QMetaObject.connectSlotsByName(MyApp)

    def retranslateUi(self, MyApp):
        _translate = QtCore.QCoreApplication.translate
        MyApp.setWindowTitle(_translate("MyApp", "Automatic License Plate Recognition"))
        self.start_app_button.setText(_translate("MyApp", "Start"))
        self.stop_app_button.setText(_translate("MyApp", "Stop"))
