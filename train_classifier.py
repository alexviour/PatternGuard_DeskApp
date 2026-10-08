import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def train_model(data_dir="dataset", save_path="models/pattern_classifier.pth", epochs=15, batch_size=8, lr=0.001):
    # Ensure models directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Image Transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset folder '{data_dir}' not found. Please create it and add subclass folders.")
        
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    num_classes = len(dataset.classes)
    print(f"Loaded dataset with {len(dataset)} images across {num_classes} classes: {dataset.classes}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Backbone: MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting Training...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    # Save checkpoint along with class mapping
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'num_classes': num_classes
    }
    torch.save(checkpoint, save_path)
    print(f"\nModel training complete! Saved to '{save_path}'.")

if __name__ == "__main__":
    print("Initializing training routine...")
    train_model()