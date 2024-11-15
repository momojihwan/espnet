# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ASR_mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_KT(object):
    def setupUi(self, KT):
        KT.setObjectName("KT")
        KT.resize(1510, 830)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(KT.sizePolicy().hasHeightForWidth())
        KT.setSizePolicy(sizePolicy)
        KT.setStyleSheet("background-color: rgb(47,48,51)")
        self.centralwidget = QtWidgets.QWidget(KT)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(1510, 830))
        self.centralwidget.setStyleSheet("background-color: rgb(47,48,51);\n"
"font: 10pt \"Malgun Gothic\";s")
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1684, 876))
        self.widget.setMinimumSize(QtCore.QSize(1600, 876))
        self.widget.setObjectName("widget")
        self.audio_widget = QtWidgets.QWidget(self.widget)
        self.audio_widget.setGeometry(QtCore.QRect(10, 70, 1491, 311))
        self.audio_widget.setStyleSheet("border-left: 1px solid rgb(255,255,255);border-right: 1px solid rgb(255,255,255);border-top: 1px solid rgb(255,255,255);border-bottom: 1px solid rgb(255,255,255);")
        self.audio_widget.setObjectName("audio_widget")
        self.label_widget = QtWidgets.QLabel(self.widget)
        self.label_widget.setGeometry(QtCore.QRect(10, 390, 1491, 41))
        self.label_widget.setStyleSheet("border-left: 1px solid rgb(255,255,255);border-right: 1px solid rgb(255,255,255);border-top: 1px solid rgb(255,255,255);border-bottom: 1px solid rgb(255,255,255); color: white;")
        self.label_widget.setText("")
        self.label_widget.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_widget.setObjectName("label_widget")
        self.asml_logo = QtWidgets.QWidget(self.widget)
        self.asml_logo.setEnabled(True)
        self.asml_logo.setGeometry(QtCore.QRect(1310, 760, 1381, 51))
        self.asml_logo.setMinimumSize(QtCore.QSize(40, 0))
        self.asml_logo.setMaximumSize(QtCore.QSize(90, 16777215))
        self.asml_logo.setStyleSheet("image: url(./Pictures/asml.png);\n"
"border: 0px;")
        self.asml_logo.setObjectName("asml_logo")
        self.play_button = QtWidgets.QPushButton(self.widget)
        self.play_button.setGeometry(QtCore.QRect(1300, 10, 50, 50))
        self.play_button.setMinimumSize(QtCore.QSize(50, 50))
        self.play_button.setStyleSheet("image: url(./Pictures/play_clicked.png);\n"
"border: 0px;")
        self.play_button.setText("")
        self.play_button.setObjectName("play_button")
        self.stop_button = QtWidgets.QPushButton(self.widget)
        self.stop_button.setGeometry(QtCore.QRect(1380, 10, 50, 50))
        self.stop_button.setMinimumSize(QtCore.QSize(50, 50))
        self.stop_button.setStyleSheet("image: url(./Pictures/stop_clicked.png);\n"
"border: 0px;")
        self.stop_button.setText("")
        self.stop_button.setObjectName("stop_button")
        self.labname_label = QtWidgets.QLabel(self.widget)
        self.labname_label.setGeometry(QtCore.QRect(1390, 750, 1521, 61))
        font = QtGui.QFont()
        font.setFamily("Malgun Gothic")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.labname_label.setFont(font)
        self.labname_label.setObjectName("labname_label")
        self.labname_label.setStyleSheet("color: rgb(255,255,255)")
        self.FileList = QtWidgets.QComboBox(self.widget)
        self.FileList.setGeometry(QtCore.QRect(13, 20, 961, 31))
        self.FileList.setObjectName("FileList")
        self.pred_widget = QtWidgets.QLabel(self.widget)
        self.pred_widget.setGeometry(QtCore.QRect(10, 440, 1491, 301))
        self.pred_widget.setStyleSheet("border-left: 1px solid rgb(255,255,255);border-right: 1px solid rgb(255,255,255);border-top: 1px solid rgb(255,255,255);border-bottom: 1px solid rgb(255,255,255); color: white; font-size: 10pt;")
        self.pred_widget.setText("")
        self.pred_widget.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop|QtCore.Qt.AlignCenter)
        self.pred_widget.setObjectName("pred_widget")
        self.hy_logo = QtWidgets.QWidget(self.widget)
        self.hy_logo.setEnabled(True)
        self.hy_logo.setGeometry(QtCore.QRect(0, 750, 221, 71))
        self.hy_logo.setMinimumSize(QtCore.QSize(40, 0))
        self.hy_logo.setMaximumSize(QtCore.QSize(10000, 16777215))
        self.hy_logo.setStyleSheet("image: url(./Pictures/hy.png);\n"
"border: 0px;")
        self.hy_logo.setObjectName("hy_logo")
        KT.setCentralWidget(self.centralwidget)

        self.retranslateUi(KT)
        QtCore.QMetaObject.connectSlotsByName(KT)

    def retranslateUi(self, KT):
        _translate = QtCore.QCoreApplication.translate
        KT.setWindowTitle(_translate("KT", "Automatic Speech Recognition"))
        self.labname_label.setText(_translate("KT", "한양대학교\n"
"음성음향신호처리\n"
"머신러닝연구실"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    KT = QtWidgets.QMainWindow()
    ui = Ui_KT()
    ui.setupUi(KT)
    KT.show()
    sys.exit(app.exec_())