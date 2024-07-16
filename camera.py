import os
import random
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
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PIL import Image
import numpy as np

# Path for model file
checkpoint_path = './dtd_resnet50_model.pth'

class DTDModel(nn.Module):
  def __init__(self, num_classes=47):
    super(DTDModel, self).__init__()
    self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

  def forward(self, x):
    return self.model(x)

# Creates model and loads file
model = DTDModel()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform for image data
transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# List of texture / pattern names
class_names = sorted([
  'banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed',
  'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked',
  'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced',
  'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley',
  'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly',
  'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded',
  'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged'
])

# Main application class
class CameraFeed(QWidget):
  def __init__(self):
    super().__init__()

    # Determines which camera is used (for Mac)
    # 0 to connect to iPhone camera, 1 to use computer camera
    self.camera = 0

    self.frame_counter = 0
    self.pred_interval = 100

    # Width and height of application
    self.width = 1280
    self.height = 720

    # Strings for image prediction
    self.prediction = ''
    self.prev_pred = ''

    # Defines boundaries
    # Square used to crop camera image to fit to 224 x 224 for ml model
    self.surface_frame_size = QRect(int(self.width / 2 - 112), int(self.height / 2 - 112), 224, 224)
    # Boundary for square displayed on camera feed
    self.boundary_rect = QRect(int(self.width / 2 - 115), int(self.height / 2 - 115), 230, 230)

    # Sets up layout and initializes GUI elements
    self.image_label = QLabel(self)
    self.vbox = QVBoxLayout()
    self.vbox.addWidget(self.image_label)
    self.setLayout(self.vbox)

    self.hbox1 = QHBoxLayout()
    self.vbox.addLayout(self.hbox1)

    self.pred_label = QLabel("Texture: ", self)
    self.pred_label.setFont(QFont('Arial', 24))
    self.hbox1.addWidget(self.pred_label)
    
    self.frame_label = QLabel(f"Next Prediction: {self.frame_counter}/{self.pred_interval}")
    self.frame_label.setFont(QFont('Arial', 24))
    self.hbox1.addWidget(self.frame_label)

    self.hbox2 = QHBoxLayout()
    self.vbox.addLayout(self.hbox2)

    self.surface_label = QLabel(self)
    self.hbox2.addWidget(self.surface_label)

    self.example_path = './dtd/images/'
    self.example_label = QLabel(self)
    self.hbox2.addWidget(self.example_label)

    self.cap = cv2.VideoCapture(self.camera)
    self.timer  = QTimer()
    self.timer.timeout.connect(self.update_frame)
    self.timer.start(30)

  def update_frame(self):
    ret, frame = self.cap.read()
    if ret:
      # If computer camera is used, mirrors image
      if self.camera == 1:
        frame = cv2.flip(frame, 1)

      # Manages size of camera feed
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

      # Converts camera feed to QImage element
      resized_frame = cv2.resize(cropped_frame, (self.width, self.height))
      resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
      image = QImage(resized_frame, resized_frame.shape[1], resized_frame.shape[0], resized_frame.strides[0], QImage.Format_RGB888)
      
      # Draws boundary on camera feed
      frame_border = QPainter(image)
      pen = QPen(QColor(0, 255, 0))
      pen.setWidth(5)
      frame_border.setPen(pen)
      frame_border.drawRect(self.boundary_rect)
      frame_border.end()

      self.image_label.setPixmap(QPixmap.fromImage(image))
  
      self.frame_label.setText(f"Next Prediction: {self.frame_counter}/{self.pred_interval}")

      # Checks if its time for new prediction
      if self.frame_counter % self.pred_interval == 0:
        
        # Captures frame from camera feed, converts it to PIL image, and transforms it to pass to ML model
        surface_image = image.copy(self.surface_frame_size)
        self.surface_label.setPixmap(QPixmap.fromImage(surface_image))
        surface_image = qimage_to_pil_image(surface_image)
        surface_image = transform(surface_image).unsqueeze(0)

        # Prediction
        with torch.no_grad():
          outputs = model(surface_image)
          _, preds = torch.max(outputs, 1)
          self.prev_pred = self.prediction
          self.prediction = class_names[preds[0]]
      
        self.pred_label.setText(f"Texture: {self.prediction}")

        # Checks if new prediction is different from previous
        if self.prev_pred != self.prediction or self.prev_pred == '':
          # Takes random image from dataset of the prediction to provide an example
          self.example_path = './dtd/images/' + self.prediction + ''
          all_files = os.listdir(self.example_path)
          image_files = [file for file in all_files if file.lower().endswith(('.jpg'))]
          selected_image = random.choice(image_files)
          image_path = os.path.join(self.example_path, selected_image)
          self.example_label.setFixedHeight(224)
          self.example_label.setPixmap(QPixmap(image_path))
      
      if self.frame_counter == self.pred_interval:
        # Resets frame counter
        self.frame_counter = 0
      else:
        # Increments frame counter
        self.frame_counter += 1


  def closeEvent(self, event):
    self.cap.release()

# Convert from QImage to PIL image for PyTorch transform
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

# Initializes and runs application
app = QApplication(sys.argv)
window = CameraFeed()
window.setFixedSize(window.width, window.height)
window.show()
sys.exit(app.exec_())