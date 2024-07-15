import sys
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import cv2
from PyQt5.QtCore import QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PIL import Image
import numpy as np

checkpoint_path = './dtd_resnet50_model.pth'

class DTDModel(nn.Module):
  def __init__(self, num_classes=47):
    super(DTDModel, self).__init__()
    self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

  def forward(self, x):
    return self.model(x)

model = DTDModel()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = sorted([
  'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed',
  'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked',
  'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced',
  'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley',
  'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly',
  'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded',
  'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'
])

class CameraFeed(QWidget):
  def __init__(self):
    super().__init__()

    self.camera = 0

    self.width = 1280
    self.height = 720

    self.surface_frame_size = QRect(int(self.width / 2 - 112), int(self.height / 2 - 112), 224, 224)
    self.boundary_rect = QRect(int(self.width / 2 - 115), int(self.height / 2 - 115), 230, 230)

    self.image_label = QLabel(self)
    self.vbox = QVBoxLayout()
    self.vbox.addWidget(self.image_label)
    self.setLayout(self.vbox)

    self.pred_label = QLabel("Texture: ", self)
    self.pred_label.setFont(QFont('Arial', 24))
    self.vbox.addWidget(self.pred_label)

    self.surface_label = QLabel(self)
    self.vbox.addWidget(self.surface_label)

    self.cap = cv2.VideoCapture(self.camera)
    self.timer  = QTimer()
    self.timer.timeout.connect(self.update_frame)
    self.timer.start(30)

  def update_frame(self):
    ret, frame = self.cap.read()
    if ret:
      if self.camera == 1:
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
      
      frame_border = QPainter(image)
      pen = QPen(QColor(0, 255, 0))
      pen.setWidth(5)
      frame_border.setPen(pen)
      frame_border.drawRect(self.boundary_rect)
      frame_border.end()

      self.image_label.setPixmap(QPixmap.fromImage(image))

      surface_image = image.copy(self.surface_frame_size)
      self.surface_label.setPixmap(QPixmap.fromImage(surface_image))

      surface_image = qimage_to_pil_image(surface_image)
      surface_image = transform(surface_image).unsqueeze(0)

      with torch.no_grad():
        outputs = model(surface_image)
        _, preds = torch.max(outputs, 1)
        prediction = class_names[preds[0]]
      
      self.pred_label.setText(f"Texture: {prediction}")

  def closeEvent(self, event):
    self.cap.release()

def qimage_to_pil_image(qimage):
    # Convert QImage to a format that can be processed by numpy
    qimage = qimage.convertToFormat(QImage.Format.Format_RGB32)
    width = qimage.width()
    height = qimage.height()

    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # 4 for RGBA format

    # Remove alpha channel
    arr = arr[:, :, :3]

    # Convert the numpy array to a PIL Image
    pil_image = Image.fromarray(arr, 'RGB')
    return pil_image

app = QApplication(sys.argv)
window = CameraFeed()
window.setFixedSize(window.width, window.height)
window.show()
sys.exit(app.exec_())