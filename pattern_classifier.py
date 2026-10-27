from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class PatternClassifier:
    MIN_SUPPORTED_CONFIDENCE = 80.0

    def __init__(
        self,
        model_path=None,
        gate_model_path=None,
    ):
        self.device = torch.device("cpu")
        model_dir = Path(__file__).resolve().parent / "models"
        model_path = (
            Path(model_path)
            if model_path
            else model_dir / "pattern_classifier.pth"
        )
        gate_model_path = (
            Path(gate_model_path)
            if gate_model_path
            else model_dir / "textile_input_gate.pth"
        )
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

        gate_checkpoint = torch.load(
            gate_model_path, map_location=self.device
        )
        self.gate_classes = gate_checkpoint.get("classes")
        if self.gate_classes != ["non_textile", "textile"]:
            raise ValueError(
                "The textile input gate checkpoint must define "
                "['non_textile', 'textile'] classes."
            )
        self.textile_class_index = self.gate_classes.index("textile")
        self.textile_probability_threshold = float(
            gate_checkpoint["textile_threshold"]
        )
        if not 0.0 <= self.textile_probability_threshold <= 1.0:
            raise ValueError(
                "The textile input gate threshold must be between 0 and 1."
            )

        self.input_gate = models.mobilenet_v3_small(weights=None)
        gate_num_ftrs = self.input_gate.classifier[3].in_features
        self.input_gate.classifier[3] = nn.Linear(gate_num_ftrs, 2)
        self.input_gate.load_state_dict(gate_checkpoint["model_state_dict"])
        self.input_gate.to(self.device)
        self.input_gate.eval()

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
            gate_outputs = self.input_gate(tensor)
            gate_probabilities = torch.nn.functional.softmax(
                gate_outputs[0], dim=0
            )
            textile_probability = float(
                gate_probabilities[self.textile_class_index].item()
            )
            gate_confidence = textile_probability * 100.0
            gate_threshold = self.textile_probability_threshold * 100.0
            if textile_probability < self.textile_probability_threshold:
                return {
                    "main_category": "Unsupported object",
                    "subclass": "No textile or repeating fabric pattern detected",
                    "confidence": gate_confidence,
                    "is_supported": False,
                    "classification_supported": False,
                    "reason": "non_textile",
                    "support_threshold": gate_threshold,
                    "gate_confidence": gate_confidence,
                    "gate_threshold": gate_threshold,
                }

            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probabilities, dim=0)

        raw_class = self.classes[pred_idx.item()]
        confidence = float(conf.item()) * 100

        if confidence < self.MIN_SUPPORTED_CONFIDENCE:
            return {
                "main_category": "Textile or repeating pattern",
                "subclass": "Specific type not identified",
                "confidence": confidence,
                "is_supported": True,
                "classification_supported": False,
                "reason": "low_class_confidence",
                "support_threshold": self.MIN_SUPPORTED_CONFIDENCE,
                "gate_confidence": gate_confidence,
                "gate_threshold": gate_threshold,
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
            "classification_supported": True,
            "support_threshold": self.MIN_SUPPORTED_CONFIDENCE,
            "gate_confidence": gate_confidence,
            "gate_threshold": gate_threshold,
        }