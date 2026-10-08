import sys
from pattern_classifier import PatternClassifier

def test_single_image(image_path):
    print(f"Loading classifier and inspecting: {image_path}...")
    
    # Initialize classifier (loads models/pattern_classifier.pth)
    classifier = PatternClassifier()
    
    # Run prediction
    result = classifier.predict(image_path)
    
    print("\n--- Classification Result ---")
    print(f"Main Category : {result['main_category']}")
    print(f"Subclass      : {result['subclass']}")
    print(f"Confidence    : {result['confidence']:.2f}%")
    print("-----------------------------\n")

if __name__ == "__main__":
    # Pass image path as argument or set a default path here
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # Replace with an actual test image path outside your training set if available
        target_image = "dataset/binakol_kusikus/kusikus_01.png"
        
    test_single_image(target_image)