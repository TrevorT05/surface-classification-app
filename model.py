import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image

# Define the data directory
data_dir = './dtd/images'
labels_dir = './dtd/labels'
classes = os.listdir(data_dir)

# Helper function to read image paths from text files
def get_image_paths_from_file(data_dir, file_path):
    with open(file_path, 'r') as f:
        image_paths = [line.strip() for line in f]
    return [os.path.join(data_dir, img) for img in image_paths]

# Function to prepare datasets
def prepare_datasets(data_dir, labels_dir):
    train_files = []
    val_files = []
    test_files = []

    # Read train files
    for i in range(1, 11):
        train_files.extend(get_image_paths_from_file(data_dir, os.path.join(labels_dir, f'train{i}.txt')))

    # Read validation files
    for i in range(1, 11):
        val_files.extend(get_image_paths_from_file(data_dir, os.path.join(labels_dir, f'val{i}.txt')))

    # Read test files
    for i in range(1, 11):
        test_files.extend(get_image_paths_from_file(data_dir, os.path.join(labels_dir, f'test{i}.txt')))

    return train_files, val_files, test_files

# Custom dataset class
class DTDTextureDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(os.listdir(data_dir))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = os.path.basename(os.path.dirname(image_path))
        label = self.class_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Prepare the datasets
train_files, val_files, test_files = prepare_datasets(data_dir, labels_dir)

# Create the Datasets
train_dataset = DTDTextureDataset(train_files, transform=transform)
val_dataset = DTDTextureDataset(val_files, transform=transform)
test_dataset = DTDTextureDataset(test_files, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class DTDModel(nn.Module):
    def __init__(self, num_classes=47):  # DTD has 47 classes
        super(DTDModel, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

model = DTDModel()

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        
        # Save the model checkpoint if it has the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, './dtd_resnet50_best.pth')
            print(f"Model saved to ./dtd_resnet50_best.pth at epoch {epoch+1}")

# Evaluate function
def evaluate_model(model, test_loader):
    model.eval()
    corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    
    accuracy = corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')

# Train the model
num_epochs = 10
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Load the best model and evaluate it
checkpoint = torch.load('./dtd_resnet50_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
evaluate_model(model, test_loader)
