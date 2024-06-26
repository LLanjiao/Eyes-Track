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
        Form.resize(879, 614)
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
        self.fileLayout = QtWidgets.QHBoxLayout()
        self.fileLayout.setObjectName("fileLayout")
        self.openFile = QtWidgets.QPushButton(parent=Form)
        self.openFile.setObjectName("openFile")
        self.fileLayout.addWidget(self.openFile)
        self.stop = QtWidgets.QPushButton(parent=Form)
        self.stop.setObjectName("stop")
        self.fileLayout.addWidget(self.stop)
        self.locate = QtWidgets.QPushButton(parent=Form)
        self.locate.setObjectName("locate")
        self.fileLayout.addWidget(self.locate)
        self.gridLayout.addLayout(self.fileLayout, 3, 2, 1, 1)
        self.viewLayout = QtWidgets.QVBoxLayout()
        self.viewLayout.setObjectName("viewLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.leftBlink = QtWidgets.QLabel(parent=Form)
        self.leftBlink.setStyleSheet("border: 1px solid black;")
        self.leftBlink.setObjectName("leftBlink")
        self.horizontalLayout_2.addWidget(self.leftBlink)
        self.rightBlink = QtWidgets.QLabel(parent=Form)
        self.rightBlink.setStyleSheet("border: 1px solid black;")
        self.rightBlink.setObjectName("rightBlink")
        self.horizontalLayout_2.addWidget(self.rightBlink)
        self.viewLayout.addLayout(self.horizontalLayout_2)
        self.eyesImage = QtWidgets.QHBoxLayout()
        self.eyesImage.setObjectName("eyesImage")
        self.eyeImage_left = QtWidgets.QLabel(parent=Form)
        self.eyeImage_left.setStyleSheet("border: 1px solid black;")
        self.eyeImage_left.setObjectName("eyeImage_left")
        self.eyesImage.addWidget(self.eyeImage_left)
        self.eyeImage_right = QtWidgets.QLabel(parent=Form)
        self.eyeImage_right.setStyleSheet("border: 1px solid black;")
        self.eyeImage_right.setObjectName("eyeImage_right")
        self.eyesImage.addWidget(self.eyeImage_right)
        self.viewLayout.addLayout(self.eyesImage)
        self.redWeight = QtWidgets.QHBoxLayout()
        self.redWeight.setObjectName("redWeight")
        self.redWeight_left = QtWidgets.QLabel(parent=Form)
        self.redWeight_left.setStyleSheet("border: 1px solid black;")
        self.redWeight_left.setObjectName("redWeight_left")
        self.redWeight.addWidget(self.redWeight_left)
        self.redWeight_right = QtWidgets.QLabel(parent=Form)
        self.redWeight_right.setStyleSheet("border: 1px solid black;")
        self.redWeight_right.setObjectName("redWeight_right")
        self.redWeight.addWidget(self.redWeight_right)
        self.viewLayout.addLayout(self.redWeight)
        self.histogram = QtWidgets.QHBoxLayout()
        self.histogram.setObjectName("histogram")
        self.histogram_left = QtWidgets.QLabel(parent=Form)
        self.histogram_left.setStyleSheet("border: 1px solid black;")
        self.histogram_left.setObjectName("histogram_left")
        self.histogram.addWidget(self.histogram_left)
        self.histogram_right = QtWidgets.QLabel(parent=Form)
        self.histogram_right.setStyleSheet("border: 1px solid black;")
        self.histogram_right.setObjectName("histogram_right")
        self.histogram.addWidget(self.histogram_right)
        self.viewLayout.addLayout(self.histogram)
        self.binary = QtWidgets.QHBoxLayout()
        self.binary.setObjectName("binary")
        self.binary_left = QtWidgets.QLabel(parent=Form)
        self.binary_left.setStyleSheet("border: 1px solid black;")
        self.binary_left.setObjectName("binary_left")
        self.binary.addWidget(self.binary_left)
        self.binary_right = QtWidgets.QLabel(parent=Form)
        self.binary_right.setStyleSheet("border: 1px solid black;")
        self.binary_right.setObjectName("binary_right")
        self.binary.addWidget(self.binary_right)
        self.viewLayout.addLayout(self.binary)
        self.tracking = QtWidgets.QHBoxLayout()
        self.tracking.setObjectName("tracking")
        self.tracking_left = QtWidgets.QLabel(parent=Form)
        self.tracking_left.setStyleSheet("border: 1px solid black;")
        self.tracking_left.setObjectName("tracking_left")
        self.tracking.addWidget(self.tracking_left)
        self.tracking_right = QtWidgets.QLabel(parent=Form)
        self.tracking_right.setStyleSheet("border: 1px solid black;")
        self.tracking_right.setObjectName("tracking_right")
        self.tracking.addWidget(self.tracking_right)
        self.viewLayout.addLayout(self.tracking)
        self.gridLayout.addLayout(self.viewLayout, 0, 3, 1, 1)
        self.threshSlider = QtWidgets.QSlider(parent=Form)
        self.threshSlider.setProperty("value", 50)
        self.threshSlider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.threshSlider.setObjectName("threshSlider")
        self.gridLayout.addWidget(self.threshSlider, 0, 0, 1, 1)
        self.cameraLayout = QtWidgets.QHBoxLayout()
        self.cameraLayout.setObjectName("cameraLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cameraChoose = QtWidgets.QComboBox(parent=Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cameraChoose.sizePolicy().hasHeightForWidth())
        self.cameraChoose.setSizePolicy(sizePolicy)
        self.cameraChoose.setEditable(False)
        self.cameraChoose.setMaxVisibleItems(10)
        self.cameraChoose.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAtBottom)
        self.cameraChoose.setIconSize(QtCore.QSize(16, 16))
        self.cameraChoose.setObjectName("cameraChoose")
        self.cameraChoose.addItem("")
        self.cameraChoose.addItem("")
        self.horizontalLayout.addWidget(self.cameraChoose)
        self.camera = QtWidgets.QPushButton(parent=Form)
        self.camera.setObjectName("camera")
        self.horizontalLayout.addWidget(self.camera)
        self.cameraLayout.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.cameraLayout, 1, 2, 1, 1)
        self.trackMethod = QtWidgets.QGroupBox(parent=Form)
        self.trackMethod.setObjectName("trackMethod")
        self.useHoughCircles = QtWidgets.QRadioButton(parent=self.trackMethod)
        self.useHoughCircles.setGeometry(QtCore.QRect(100, 0, 150, 20))
        self.useHoughCircles.setChecked(True)
        self.useHoughCircles.setObjectName("useHoughCircles")
        self.useOperator = QtWidgets.QRadioButton(parent=self.trackMethod)
        self.useOperator.setGeometry(QtCore.QRect(250, 0, 150, 20))
        self.useOperator.setObjectName("useOperator")
        self.gridLayout.addWidget(self.trackMethod, 3, 3, 1, 1)
        self.binaryMethod = QtWidgets.QGroupBox(parent=Form)
        self.binaryMethod.setObjectName("binaryMethod")
        self.useDirectCompare = QtWidgets.QRadioButton(parent=self.binaryMethod)
        self.useDirectCompare.setGeometry(QtCore.QRect(100, 0, 150, 20))
        self.useDirectCompare.setChecked(True)
        self.useDirectCompare.setObjectName("useDirectCompare")
        self.useCoarsePositioning = QtWidgets.QRadioButton(parent=self.binaryMethod)
        self.useCoarsePositioning.setGeometry(QtCore.QRect(250, 0, 150, 20))
        self.useCoarsePositioning.setChecked(False)
        self.useCoarsePositioning.setObjectName("useCoarsePositioning")
        self.gridLayout.addWidget(self.binaryMethod, 1, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.faceFrame.setText(_translate("Form", "faceFrame"))
        self.openFile.setText(_translate("Form", "openFile"))
        self.stop.setText(_translate("Form", "stop"))
        self.locate.setText(_translate("Form", "locate"))
        self.leftBlink.setText(_translate("Form", "leftBlink"))
        self.rightBlink.setText(_translate("Form", "rightBlink"))
        self.eyeImage_left.setText(_translate("Form", "eyeImage_left"))
        self.eyeImage_right.setText(_translate("Form", "eyeImage_right"))
        self.redWeight_left.setText(_translate("Form", "redWeight_left"))
        self.redWeight_right.setText(_translate("Form", "redWeight_right"))
        self.histogram_left.setText(_translate("Form", "histogram_left"))
        self.histogram_right.setText(_translate("Form", "histogram_right"))
        self.binary_left.setText(_translate("Form", "binary_left"))
        self.binary_right.setText(_translate("Form", "binary_right"))
        self.tracking_left.setText(_translate("Form", "tracking_left"))
        self.tracking_right.setText(_translate("Form", "tracking_right"))
        self.cameraChoose.setCurrentText(_translate("Form", "computerCamera"))
        self.cameraChoose.setItemText(0, _translate("Form", "computerCamera"))
        self.cameraChoose.setItemText(1, _translate("Form", "phoneCamera"))
        self.camera.setText(_translate("Form", "openCamera"))
        self.trackMethod.setTitle(_translate("Form", "trackMethod"))
        self.useHoughCircles.setText(_translate("Form", "useHoughCircles"))
        self.useOperator.setText(_translate("Form", "useOperator"))
        self.binaryMethod.setTitle(_translate("Form", "binaryMethod"))
        self.useDirectCompare.setText(_translate("Form", "useDirectCompare"))
        self.useCoarsePositioning.setText(_translate("Form", "useCoarsePositioning"))
