# Smart Fridge – AI Food Detection & Shelf-Life Assistant

An AI-powered smart-fridge application that combines image classification, live camera input, voice feedback, shelf-life estimation, and recipe suggestions in a Tkinter desktop interface.

## Features

- Food image classification using **MobileNetV2 transfer learning**
- Image upload and prediction through a Tkinter GUI
- Live camera detection through an IP Webcam stream
- Automatic image capture from the live stream
- Voice feedback using **pyttsx3**
- Temperature and humidity based shelf-life classification
- Recipe suggestions based on detected food and shelf-life category
- Reusable trained model saved as `mobilenet_food_model.h5`

## Technology Stack

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Tkinter
- Pillow
- pyttsx3

## Model Configuration

The submitted project code uses:

- Image size: `150 × 150`
- Batch size: `32`
- Training epochs: `3`
- MobileNetV2 base model with ImageNet weights
- Frozen MobileNetV2 feature extractor
- GlobalAveragePooling2D
- Dropout: `0.3`
- Dense layer: `128` ReLU units
- Softmax output based on the discovered dataset class folders

## Shelf-Life Logic

The current code classifies shelf life from the entered temperature and humidity:

| Condition | Shelf-life category | Range |
|---|---|---|
| Temperature > 30°C OR humidity > 60% | Low | 1–2 days |
| 20°C < temperature ≤ 30°C AND 40% < humidity ≤ 60% | Medium | 3–5 days |
| Otherwise | High | 5–7 days |

These thresholds are the logic implemented in the submitted project code; they should be treated as project logic rather than a food-safety standard.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sarabjit-yadav/smart-fridge-ai.git
cd smart-fridge-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Tkinter is normally included with standard Python installations on Windows.

### 3. Configure the dataset

Before training, update `train_dir` and `val_dir` in `smart_fridge.py` to the actual dataset directories. The submitted source currently contains local Windows paths, so those paths will need to be changed on another computer.

The code expects a directory structure compatible with Keras `flow_from_directory`, with class folders inside the training and validation directories.

### 4. Configure the camera

Update `VIDEO_URL` in `smart_fridge.py` with the IP Webcam video URL for your phone/camera.

The submitted code currently contains:

```text
http://192.168.1.72:4747/video
```

### 5. Run

```bash
python smart_fridge.py
```

If `mobilenet_food_model.h5` is not present, the code trains the MobileNetV2-based classifier and saves the model. If the model file already exists, the code loads it.

## GUI Functions

- **Upload Image** – select a food image and classify it.
- **Shelf Life & Recipes** – enter temperature/humidity and view shelf-life category plus available recipes.
- **Live Detection** – read frames from the configured IP camera and run prediction periodically.
- **Use Captured Image** – classify the latest image captured from live detection.
- **Clear** – clear the temperature, humidity, and displayed image.

## Repository Structure

```text
smart-fridge-ai/
├── smart_fridge.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── README.md
└── documentation/
    └── system-overview.md
```

## Important Notes

- The trained `.h5` model is ignored by Git because model files can be large. See `models/README.md`.
- The project code contains local dataset and camera paths that must be configured for the target computer.
- The recipe dictionary contains many food classes; the actual prediction classes are determined from the dataset directory names.
- This repository preserves the submitted project implementation rather than silently changing its model or shelf-life logic.

## Author

**Sarabjit Kumar**  
B.Tech Electronics & Communication Engineering  
Lovely Professional University
