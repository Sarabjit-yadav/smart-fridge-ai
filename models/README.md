# Model Files

The application uses `mobilenet_food_model.h5` as its trained MobileNetV2-based classifier.

The project source creates this model automatically when the model file is not present. The generated model is intentionally excluded from Git using `.gitignore` because trained model binaries can be large.

If you want to distribute a trained model with the project, upload it separately using an appropriate model-storage/release mechanism and update the run instructions accordingly.
