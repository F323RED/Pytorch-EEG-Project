# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\mirror_BCI\NewGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(427, 590)
        MainWindow.setStyleSheet("border-color: rgb(85, 170, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.btnCon = QtWidgets.QPushButton(self.centralwidget)
        self.btnCon.setCheckable(False)
        self.btnCon.setObjectName("btnCon")
        self.gridLayout.addWidget(self.btnCon, 1, 0, 1, 1)
        self.btnSave = QtWidgets.QPushButton(self.centralwidget)
        self.btnSave.setObjectName("btnSave")
        self.gridLayout.addWidget(self.btnSave, 3, 0, 1, 1)
        self.texbConStatus = QtWidgets.QTextBrowser(self.centralwidget)
        self.texbConStatus.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.texbConStatus.sizePolicy().hasHeightForWidth())
        self.texbConStatus.setSizePolicy(sizePolicy)
        self.texbConStatus.setMinimumSize(QtCore.QSize(256, 0))
        self.texbConStatus.setMaximumSize(QtCore.QSize(256, 192))
        self.texbConStatus.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.texbConStatus.setObjectName("texbConStatus")
        self.gridLayout.addWidget(self.texbConStatus, 0, 0, 1, 1)
        self.btnDisCon = QtWidgets.QPushButton(self.centralwidget)
        self.btnDisCon.setObjectName("btnDisCon")
        self.gridLayout.addWidget(self.btnDisCon, 6, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("NCU.png"))
        self.label.setScaledContents(False)
        self.label.setWordWrap(False)
        self.label.setOpenExternalLinks(False)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 7, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 427, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btnCon.setText(_translate("MainWindow", "Connect"))
        self.btnSave.setText(_translate("MainWindow", "Save"))
        self.btnDisCon.setText(_translate("MainWindow", "Disconnect"))

