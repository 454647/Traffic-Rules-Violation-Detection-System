import time
import cv2
import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QListWidget, QAction, qApp, QMenu
from PyQt5.uic import loadUi
import numpy as np

from Archive import ArchiveWindow
from Database import Database
from processor.MainProcessor import MainProcessor
from processor.TrafficProcessor import TrafficProcessor
from ViolationItem import ViolationItem
from add_windows.AddCamera import AddCamera
from add_windows.AddCar import AddCar
from add_windows.AddRule import AddRule
from add_windows.AddViolation import AddViolation

from plateDetection import Main

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("./UI/MainWindow.ui", self)

        self.live_preview.setScaledContents(True)
        from PyQt5.QtWidgets import QSizePolicy
        self.live_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.cam_clear_guard = False
        self.allow_snap = False
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Welcome")

        self.clear_button.clicked.connect(self.clear)
        self.refresh_button.clicked.connect(self.refresh)

        self.database = Database.getInstance()
        self.database.deleteAllCars()
        # self.database.deleteAllViolations()

        cam_groups = self.database.getCamGroupList()
        self.camera_group.clear()
        self.camera_group.addItems(name for name in cam_groups)
        self.camera_group.setCurrentIndex(0)
        self.camera_group.currentIndexChanged.connect(self.camGroupChanged)

        cams = self.database.getCamList(self.camera_group.currentText())
        self.cam_selector.clear()
        self.cam_selector.addItems(name for name, location, feed in cams)
        self.cam_selector.setCurrentIndex(0)
        self.cam_selector.currentIndexChanged.connect(self.camChanged)
        self.take_plate_snap.clicked.connect(self.take_snap)
        self.processor = MainProcessor(self.cam_selector.currentText())

        self.violation_list = self.listWidget

        self.feed = None
        self.vs = None
        self.updateCamInfo()

        self.updateLog()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)

        self.car_cascade = cv2.CascadeClassifier('cars.xml')

        # trafficLightTimer = QTimer(self)
        # trafficLightTimer.timeout.connect(self.toggleLight)
        # trafficLightTimer.start(5000)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_G:
            self.processor.setLight("Green")
        elif event.key() == QtCore.Qt.Key_R:
            self.processor.setLight("Red")
        elif event.key() == QtCore.Qt.Key_S:
            self.toggleLight()

    def take_snap(self):
        self.allow_snap = True

    minx = 1000
    maxx = 0

    def update_image(self):
        _, frame = self.vs.read()
        if frame is None:
            self.updateCamInfo()
            self.updateLog()
            return

        # packet = self.processor.getProcessedImage(frame)
        blurrend = cv2.GaussianBlur(frame, (5, 5), 0)
        gray = cv2.cvtColor(blurrend, cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
        edges = cv2.Canny(blur_gray, 50, 150, 3)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=15, maxLineGap=5)
        for x1, y1, x2, y2 in lines[0]:
            if y1 > 150 and y2 > 150:
                self.minx = min(self.minx, x1)
                self.maxx = max(self.maxx, x2)
                cv2.line(frame, (self.minx, y1), (self.maxx, y2), (255, 0, 0), 3)
        cars = self.car_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in cars:
            if w > 140 and h > 140:
                cv2.rectangle(frame, (x+20, y), (x + w, y + h-80), (0, 255, 0), 2)

        if self.allow_snap:
            self.allow_snap = False
            plate_img, plate = Main.main(frame)
            carId = self.database.getMaxCarId() + 1
            car_img = 'car_' + str(carId) + '.png'
            cv2.imwrite('car_images/' + car_img, plate_img)
            if plate:
                self.database.insertIntoCars(car_id=carId, car_img=car_img , lic_img=plate, lic_num=plate)
                self.database.insertIntoViolations(camera=self.cam_selector.currentText(), car=carId, rule='1',
                                                   time=time.time())
            else:
                self.database.insertIntoCars(car_id=carId, car_img=car_img)
                self.database.insertIntoViolations(camera=self.cam_selector.currentText(), car=carId, rule='1',
                                                   time=time.time())
            self.updateLog()
        try:
            qimg = self.toQImage(frame)
            self.live_preview.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            print("unable to draw")

    def updateCamInfo(self):
        count, location, self.feed = self.database.getCamDetails(self.cam_selector.currentText())
        self.feed = 'videos/' + self.feed
        self.processor = MainProcessor(self.cam_selector.currentText())
        self.vs = cv2.VideoCapture(self.feed)
        # self.cam_id.setText(self.cam_selector.currentText())
        # self.address.setText(location)
        # self.total_records.setText(str(count))

    def updateLog(self):
        self.violation_list.clear()
        rows = self.database.getViolationsFromCam(str(self.cam_selector.currentText()))
        for row in rows:
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.violation_list)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.violation_list.addItem(listWidgetItem)
            self.violation_list.setItemWidget(listWidgetItem, listWidget)

    @QtCore.pyqtSlot()
    def refresh(self):
        self.updateCamInfo()
        self.updateLog()

    @QtCore.pyqtSlot()
    def search(self):
        from SearchWindow import SearchWindow
        searchWindow = SearchWindow(self.search_result, parent=self)
        searchWindow.show()

    @QtCore.pyqtSlot()
    def clear(self):
        qm = QtWidgets.QMessageBox
        prompt = qm.question(self, '', "Are you sure to reset all the values?", qm.Yes | qm.No)
        if prompt == qm.Yes:
            self.database.clearCamLog()
            self.database.deleteAllViolations()
            self.updateLog()
        else:
            pass

    def toQImage(self, raw_img):
        from numpy import copy
        img = copy(raw_img)
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img.tobytes(), img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()
        return outImg

    @QtCore.pyqtSlot()
    def camChanged(self):
        if not self.cam_clear_guard:
            self.updateCamInfo()
            self.updateLog()

    @QtCore.pyqtSlot()
    def camGroupChanged(self):
        cams = self.database.getCamList(self.camera_group.currentText())
        self.cam_clear_guard = True
        self.cam_selector.clear()
        self.cam_selector.addItems(name for name, location, feed in cams)
        self.cam_selector.setCurrentIndex(0)
        # self.cam_selector.currentIndexChanged.connect(self.camChanged)
        self.cam_clear_guard = False
        self.updateCamInfo()
