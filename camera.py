import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

class CameraFeed(QWidget):
  def __init__(self):
    super().__init__()

    self.width = 640
    self.height = 480

    self.image_label = QLabel(self)
    self.vbox = QVBoxLayout()
    self.vbox.addWidget(self.image_label)
    self.setLayout(self.vbox)

    self.cap = cv2.VideoCapture(1)
    self.timer  = QTimer()
    self.timer.timeout.connect(self.update_frame)
    self.timer.start(30)

  def update_frame(self):
    ret, frame = self.cap.read()
    if ret:
      frame = cv2.flip(frame, 1)

      frame_height, frame_width = frame.shape[:2]

      window_aspect_ratio = self.width / self.height
      frame_aspect_ratio = frame_width / frame_height

      if frame_aspect_ratio > window_aspect_ratio:
        new_width = int(frame_height * window_aspect_ratio)
        offset = (frame_width - new_width) // 2
        cropped_frame = frame[:, offset:offset + new_width]
      else:
        new_height = int(frame_width / window_aspect_ratio)
        offset = (frame_height - new_height) // 2
        cropped_frame = frame[offset:offset + new_height, :]

      resized_frame = cv2.resize(cropped_frame, (self.width, self.height))
      resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
      image = QImage(resized_frame, resized_frame.shape[1], resized_frame.shape[0], resized_frame.strides[0], QImage.Format_RGB888)
      self.image_label.setPixmap(QPixmap.fromImage(image))

  def closeEvent(self, event):
    self.cap.release()

app = QApplication(sys.argv)
window = CameraFeed()
window.setFixedSize(window.width, window.height)
window.show()
sys.exit(app.exec_())