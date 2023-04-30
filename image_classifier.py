import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from collections import defaultdict

# Load the pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Set the path to the folder containing the images
folder_path = r'C:\Users\aalon\OneDrive\Pictures\iCloud Photos from Danielle Hildebrand'

# Initialize the seen images dictionary
seen_images = defaultdict(float)

# Get the image files
image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp')]

# Initialize the progress counter
total_files = len(image_files)
processed_files = 0

# Loop over the files in the folder
for filename in image_files:
    # Load and preprocess the image
    img_path = os.path.join(folder_path, filename)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except:
        print(f"Error: Could not load image file {filename}")
        continue
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Make a prediction
    preds = model.predict(x)
    labels = decode_predictions(preds, top=1)[0]
    label, description, confidence = labels[0]

    # Check if the predicted image has already been seen
    image_key = (label, description)
    if seen_images[image_key] == 0:
        seen_images[image_key] = 1.0
    else:
        seen_images[image_key] += 0.1
    image_counter = seen_images[image_key]

    # Rename the file with the predicted label, confidence, and image counter
    new_filename = f"{confidence:.2%}_{description}_{image_counter:.1f}{os.path.splitext(filename)[1]}"
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

    # Update the progress counter
    processed_files += 1
    print(f"Processed {processed_files}/{total_files} files.")

print("Finished processing all files.")
