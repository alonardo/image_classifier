# Image Classifier

This repository contains a Python script, `image_classifier.py`, which uses a pre-trained ResNet50 model to classify images in a specified folder. The script renames the image files with the predicted class, confidence score, and a counter for each unique item found, starting with 1.0 and incrementing by 0.1 for each unique item.

## Dependencies

- Python 3.6 or later
- TensorFlow 2.x
- NumPy
- Pillow (PIL)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your_username/image_classifier.git

pip install -r requirements.txt
python image_classifier.py
