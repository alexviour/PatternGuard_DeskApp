import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class PatternClassifier:
    MIN_SUPPORTED_CONFIDENCE = 80.0

    def __init__(self, model_path="models/pattern_classifier.pth"):
        self.device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Safe fallback for class names key
        if 'classes' in checkpoint:
            self.classes = checkpoint['classes']
        elif 'class_to_idx' in checkpoint:
            self.classes = list(checkpoint['class_to_idx'].keys())
        else:
            raise KeyError("Could not find class list or class mapping in checkpoint.")

        # Re-build MobileNetV3-Small architecture
        self.model = models.mobilenet_v3_small(weights=None)
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, len(self.classes))
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Fast CPU preprocess transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probabilities, dim=0)

        raw_class = self.classes[pred_idx.item()]
        confidence = float(conf.item()) * 100

        if confidence < self.MIN_SUPPORTED_CONFIDENCE:
            return {
                "main_category": "Unsupported or uncertain input",
                "subclass": "Not recognized as Pinilian or Binakol",
                "confidence": confidence,
                "is_supported": False,
                "support_threshold": self.MIN_SUPPORTED_CONFIDENCE,
            }
        
        # Parse category and subclass from folder name string
        if raw_class.startswith("binakol"):
            main_cat = "Binakol"
            subclass = raw_class.replace("binakol_", "").replace("_", " ").title()
        else:
            main_cat = "Pinilian"
            subclass = raw_class.replace("pinilian_", "").replace("_", " ").title()

        return {
            "main_category": main_cat,
            "subclass": subclass,
            "confidence": confidence,
            "is_supported": True,
            "support_threshold": self.MIN_SUPPORTED_CONFIDENCE,
        }