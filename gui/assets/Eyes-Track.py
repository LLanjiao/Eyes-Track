# Form implementation generated from reading ui file 'Eyes-Track.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(850, 614)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.faceFrame = QtWidgets.QLabel(parent=Form)
        self.faceFrame.setMouseTracking(False)
        self.faceFrame.setStyleSheet("border: 1px solid black;")
        self.faceFrame.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.faceFrame.setObjectName("faceFrame")
        self.gridLayout.addWidget(self.faceFrame, 0, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.startCamera = QtWidgets.QPushButton(parent=Form)
        self.startCamera.setObjectName("startCamera")
        self.horizontalLayout_3.addWidget(self.startCamera)
        self.openFile = QtWidgets.QPushButton(parent=Form)
        self.openFile.setObjectName("openFile")
        self.horizontalLayout_3.addWidget(self.openFile)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 2, 1, 1)
        self.threshSlider = QtWidgets.QSlider(parent=Form)
        self.threshSlider.setProperty("value", 50)
        self.threshSlider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.threshSlider.setObjectName("threshSlider")
        self.gridLayout.addWidget(self.threshSlider, 0, 0, 1, 1)
        self.stop = QtWidgets.QPushButton(parent=Form)
        self.stop.setObjectName("stop")
        self.gridLayout.addWidget(self.stop, 1, 3, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_10 = QtWidgets.QLabel(parent=Form)
        self.label_10.setStyleSheet("border: 1px solid black;")
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_6.addWidget(self.label_10)
        self.eyesImage = QtWidgets.QLabel(parent=Form)
        self.eyesImage.setStyleSheet("border: 1px solid black;")
        self.eyesImage.setObjectName("eyesImage")
        self.horizontalLayout_6.addWidget(self.eyesImage)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(parent=Form)
        self.label_2.setStyleSheet("border: 1px solid black;")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.redWeight = QtWidgets.QLabel(parent=Form)
        self.redWeight.setStyleSheet("border: 1px solid black;")
        self.redWeight.setObjectName("redWeight")
        self.horizontalLayout.addWidget(self.redWeight)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_4 = QtWidgets.QLabel(parent=Form)
        self.label_4.setStyleSheet("border: 1px solid black;")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.histogram = QtWidgets.QLabel(parent=Form)
        self.histogram.setStyleSheet("border: 1px solid black;")
        self.histogram.setObjectName("histogram")
        self.horizontalLayout_5.addWidget(self.histogram)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_6 = QtWidgets.QLabel(parent=Form)
        self.label_6.setStyleSheet("border: 1px solid black;")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.binary = QtWidgets.QLabel(parent=Form)
        self.binary.setStyleSheet("border: 1px solid black;")
        self.binary.setObjectName("binary")
        self.horizontalLayout_4.addWidget(self.binary)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_7 = QtWidgets.QLabel(parent=Form)
        self.label_7.setStyleSheet("border: 1px solid black;")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_2.addWidget(self.label_7)
        self.rightEye = QtWidgets.QLabel(parent=Form)
        self.rightEye.setStyleSheet("border: 1px solid black;")
        self.rightEye.setObjectName("rightEye")
        self.horizontalLayout_2.addWidget(self.rightEye)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 1, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.faceFrame.setText(_translate("Form", "faceFrame"))
        self.startCamera.setText(_translate("Form", "startCamera"))
        self.openFile.setText(_translate("Form", "openFile"))
        self.stop.setText(_translate("Form", "stop"))
        self.label_10.setText(_translate("Form", "TextLabel"))
        self.eyesImage.setText(_translate("Form", "eyesImage"))
        self.label_2.setText(_translate("Form", "TextLabel"))
        self.redWeight.setText(_translate("Form", "redWeight"))
        self.label_4.setText(_translate("Form", "TextLabel"))
        self.histogram.setText(_translate("Form", "histogram"))
        self.label_6.setText(_translate("Form", "TextLabel"))
        self.binary.setText(_translate("Form", "binary"))
        self.label_7.setText(_translate("Form", "TextLabel"))
        self.rightEye.setText(_translate("Form", "rightEye"))
