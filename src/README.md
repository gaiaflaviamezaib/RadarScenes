# Source Code Structure

Welcome to the brains of the operation! This `src/` folder contains all the Python code that makes the machine learning pipeline work. To keep things clean and easy to find, the code is split into three main areas:

## 1. Preparing the Data (`data/`)
Before we can train AI, we need clean data. These scripts do the heavy lifting:
- **`build_features.py`**: The main script. It reads the massive raw radar files and turns them into clean, organized CSV tables that our AI can actually understand.
- **`overview.py`**: A quick script that counts how many cars, pedestrians, and bikes are in the dataset.
- **`link_sensors.py`**: Matches the radar "blips" with the actual camera photos so we can see what the radar was looking at.

## 2. Training the AI (`models/`)
This is where the actual machine learning happens:
- **`train.py`**: Teaches several AI models (like Random Forest and Logistic Regression) how to recognize different objects.
- **`evaluate.py`**: Tests the models on new data to see how smart they actually are, and creates scorecards (like accuracy and F1 scores).

## 3. Creating Charts (`visualization/`)
- **`visualize_dataset.py`**: Generates beautiful charts and graphs showing how our data and models are performing.

Finally, **`utils.py`** is just a helper file containing shared code (like math formulas and formatting rules) that the other scripts borrow from so we don't repeat ourselves.
