# Vehicle Detection Project

This project trains a simple convolutional neural network to classify whether a road-region image contains a vehicle, then applies that model to frames from a video using OpenCV.

## What It Does

- Loads labeled images from `dataset/vehicles/` and `dataset/non-vehicles/`
- Resizes images to `64x64`
- Trains a binary image classifier with TensorFlow/Keras
- Loads the saved model and runs frame-by-frame inference on a video
- Displays the video with a `Vehicle Detected` overlay when the model output exceeds `0.5`

## Project Structure

```text
vehicle-detection-project/
├── dataset/
│   ├── non-vehicles/
│   └── vehicles/
├── detection/
│   ├── detect_vehicle.py
│   └── test_video.mp4
├── model/
│   └── train_model.py
├── preprocessing/
│   └── preprocess.py
├── saved_model/
│   └── vehicle_model.h5
├── utils/
│   └── feature_extraction.py
└── requirements.txt
```

## Requirements

- Python 3.9+
- pip
- A local display environment for OpenCV windows

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The training script expects two folders:

- `dataset/vehicles/`: images containing vehicles
- `dataset/non-vehicles/`: images not containing vehicles

Images are loaded recursively, resized to `64x64`, and converted from BGR to RGB during preprocessing.

## Train the Model

Run the training script from the `model` directory:

```bash
cd model
python train_model.py
```

What happens during training:

- The dataset is loaded from `../dataset/vehicles` and `../dataset/non-vehicles`
- Pixel values are normalized to the range `[0, 1]`
- Data is split into training and validation sets using an 80/20 split
- A 3-layer CNN is trained for 10 epochs
- The trained model is saved to `saved_model/vehicle_model.h5`

## Run Vehicle Detection

Run the detection script from the `detection` directory:

```bash
cd detection
python detect_vehicle.py
```

Current inference behavior:

- Loads the model from `../saved_model/vehicle_model.h5`
- Opens `detection/test_video.mp4`
- Uses the lower half of each frame as the region of interest
- Resizes the ROI to `64x64`
- Runs prediction on each frame
- Draws `Vehicle Detected` on the frame when the prediction is greater than `0.5`
- Press `Esc` to close the video window

## Notes About the Current Implementation

- `detection/detect_vehicle.py` uses a hardcoded absolute path for the video file:
  `C:/Users/usr/Desktop/vehicle-detection-project/detection/test_video.mp4`
- Because relative paths are used for the model and dataset, the scripts should be run from their current folders:
  - `model/train_model.py` from `model/`
  - `detection/detect_vehicle.py` from `detection/`
- The project currently performs image classification on a selected region of each frame. It is not a full object detector with bounding boxes.
- `utils/feature_extraction.py` exists but is not used by the current training or detection pipeline.

## Troubleshooting

If the model file is missing:

```bash
cd model
python train_model.py
```

If the OpenCV window does not open:

- Check that you are running the script in a desktop session, not a headless environment.
- Verify that `opencv-python` is installed.

If the video cannot be found:

- Confirm that `detection/test_video.mp4` exists.
- Update the path inside `detection/detect_vehicle.py` if your project is located elsewhere.

## Possible Improvements

- Replace hardcoded paths with command-line arguments or config values
- Add evaluation metrics after training
- Save training history and plots
- Support webcam input
- Use a sliding window or modern object detection model for more accurate localization
