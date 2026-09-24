# System Overview

## Input

The application accepts either:
1. A food image selected from the computer, or
2. Frames from a configured IP Webcam stream.

Temperature and humidity are entered through the Tkinter GUI for shelf-life estimation.

## AI Pipeline

The submitted implementation uses MobileNetV2 as a frozen feature extractor. A GlobalAveragePooling2D layer, Dropout layer, Dense layer, and softmax output layer are added for food-class classification.

## Detection Flow

1. Load or train the MobileNetV2-based model.
2. Load class names from the training directory.
3. Accept an uploaded image or live camera frame.
4. Resize the image to 150 × 150.
5. Normalize pixel values by dividing by 255.
6. Run model prediction.
7. Select the class with the highest predicted probability.
8. Display the predicted class and confidence.
9. Provide voice feedback.

## Shelf-Life and Recipe Flow

The user enters temperature and humidity. The code maps those values to Low, Medium, or High shelf-life categories and then looks up recipes for the detected item and category.

## Live Camera Flow

The live detection function opens the configured IP Webcam URL, performs a prediction approximately every two seconds, displays the prediction on the OpenCV window, and saves the latest frame as `captured_from_live.jpg`.

## Current Project Scope

The submitted code is a desktop AI prototype integrating computer vision, a GUI, voice output, and rule-based shelf-life/recipe assistance. Hardware temperature/humidity sensors and ESP32 integration are not implemented in this specific Python file.
