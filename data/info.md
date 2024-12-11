# Yelp Photo Classification - Data Setup Guide

## Data Download Instructions

### 1. Yelp Dataset
Download the Yelp Photos Dataset from the official Yelp Dataset Challenge:
- Visit: [Yelp Dataset Challenge](https://www.yelp.com/dataset)
- Download the "Yelp Photos Dataset."
- You will need:
  - `photos.json`: Contains metadata and labels.
  - Photo files (collection of `.jpg` images).

### 2. File Organization
Organize the files as follows:

#### Data
- `data/`
  - `yelp_photos/`
    - `photos.json` (Metadata file)
    - `photos/` (Directory containing all `.jpg` files)

#### Models (Created by running the model scripts)
- `models/`
  - `best_model.keras`
  - `best_model_with_test.keras`
  - `final_model.keras`

#### Outputs (Generated during training and evaluation)
- `outputs/`
  - `confusion_matrix.png`
  - `quick_confusion_matrix.png`
  - `training_history.png`

#### Source Code
- `src/`
  - `main.py`
  - `quick_evaluate.py`
  - `train_test_model.py`

### 3. Data Verification
- Run `main.py` to verify data distribution.
- Expected dataset size: ~200,100 images.
- Class distribution should show:
  - **Food**: ~108,047 images (54.03%).
  - **Inside**: ~56,030 images (28.02%).
  - **Outside**: ~18,569 images (9.28%).
  - **Drink**: ~15,670 images (7.84%).
  - **Menu**: ~1,678 images (0.84%).

### 4. File Paths
Ensure the following paths in the scripts are correctly set:

```python
# Path to the JSON metadata file
json_path = 'data/yelp_photos/photos.json'

# Path to the directory containing image files
PHOTO_DIR = 'data/yelp_photos/photos'
