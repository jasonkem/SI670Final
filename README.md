# Yelp Photo Classification Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Background](#dataset-background)
   - [Key Dataset Statistics](#key-dataset-statistics)
3. [Data Setup and Download](#data-setup-and-download)
4. [Technical Approach](#technical-approach)
   - [Model Architecture](#model-architecture)
   - [Training Configuration](#training-configuration)
5. [Performance Metrics](#performance-metrics)
   - [Per-Class Performance](#per-class-performance)
   - [Key Observations](#key-observations)
6. [Challenges Addressed](#challenges-addressed)
7. [Practical Implications](#practical-implications)
8. [Future Directions](#future-directions)

---

## Project Overview
The Yelp Photo Classification project aimed to tackle the challenge of organizing and understanding vast amounts of business-related visual data uploaded by users to Yelp’s platform. Yelp users share millions of photos to highlight their experiences at businesses, including images of food, drinks, interiors, menus, and building exteriors. These photos provide valuable insights to potential customers, but categorizing and presenting this data effectively at scale is a significant challenge.

Manually sorting millions of photos is infeasible, making automated classification critical for improving user experience. Accurate photo classification enables Yelp to:
- **Enhance Search Functionality**: Users can filter results by photo categories, such as "restaurants with outdoor seating" or "cafes with great interior designs."
- **Improve Recommendations**: Yelp can highlight visual features of businesses to match user preferences.
- **Streamline Content Moderation**: Automating categorization supports the organization of business photo galleries and improves the browsing experience.
- **Assist Business Owners**: Businesses can display their offerings more effectively, improving their visibility and appeal to customers.

Given the class imbalance in the dataset, with certain categories like "menu" underrepresented, and the diverse nature of user-uploaded photos (varying in quality, lighting, and angles), building a high-performing model required state-of-the-art deep learning techniques.

---

## Dataset Background

The dataset comprised 200,100 business-related images collected from Yelp. Each image was labeled into one of five categories:
- **Food**: Images of meals or dishes.
- **Inside**: Photos of a business’s interior, including seating or decor.
- **Outside**: Exterior shots of buildings or storefronts.
- **Drink**: Images of beverages, such as coffee, cocktails, or soft drinks.
- **Menu**: Photos of menus, both physical and digital.

### Key Dataset Statistics
- **Valid Images**: 199,994 (after removing 106 invalid entries using Python's PIL library for verification).
- **Class Distribution**:
  - Food: 108,047 images (54.03%)
  - Inside: 56,030 images (28.02%)
  - Outside: 18,569 images (9.28%)
  - Drink: 15,670 images (7.84%)
  - Menu: 1,678 images (0.84%)
- **Data Split**:
  - Training: 127,996 images (64%)
  - Validation: 31,999 images (16%)
  - Test: 39,999 images (20%)

The dataset’s imbalance posed challenges, especially for underrepresented categories like "menu," requiring the model to generalize well across both majority and minority classes without overfitting to dominant ones like "food."

---

## Data Setup and Download

### Download Instructions
1. Visit the [Yelp Dataset Challenge](https://www.yelp.com/dataset).
2. Download the "Yelp Photos Dataset."
3. Ensure you have the following files:
   - `photos.json`: Contains metadata and labels.
   - A collection of `.jpg` photo files.

### File Organization
Organize the files into the following structure:
- **Data**:
  - `data/yelp_photos/photos.json` (Metadata file)
  - `data/yelp_photos/photos/` (Directory containing all `.jpg` files)
- **Models** (Created by running model scripts):
  - `models/best_model.keras`
  - `models/best_model_with_test.keras`
  - `models/final_model.keras`
- **Outputs** (Generated during training and evaluation):
  - `outputs/confusion_matrix.png`
  - `outputs/quick_confusion_matrix.png`
  - `outputs/training_history.png`
- **Source Code**:
  - `src/main.py`
  - `src/quick_evaluate.py`
  - `src/train_test_model.py`

### Notes
- **Paths to Update**: Ensure these paths in the scripts match your file structure:
  ```python
  json_path = 'data/yelp_photos/photos.json'
  PHOTO_DIR = 'data/yelp_photos/photos'

# Yelp Photo Classification - Technical Details

## File Generation
- **Model Files**: `.keras` files will be generated automatically when running the training scripts.
- **Output Visualizations**: Files such as `.png` (e.g., confusion matrices and training history) will also be created during evaluation.
- **Disk Space**: Ensure ~2GB of available space for the dataset.

---

## Technical Approach

### Model Architecture
- **Base Model**: ResNet50V2 (pre-trained on ImageNet).
- **Input Shape**: 224x224 pixels, 3 channels (RGB).
- **Output Classes**: 5 (softmax activation).
- **Custom Layers**:
  - GlobalAveragePooling2D for dimensionality reduction.
  - Dense layers with ReLU activation and dropout for regularization.
  - Final dense layer for multi-class classification.

### Training Configuration
- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**:
  - Initial: 0.001
  - Reduced by 50% at epoch 8.
- **Callbacks**:
  - Early stopping to prevent overfitting.
  - Learning rate reduction on performance plateau.
  - Model checkpointing to save the best model.
- **Total Training Time**: ~12.5 hours (~75 minutes/epoch).

---

## Performance Metrics

### Overall Metrics
- **Overall Test Accuracy**: 89%
- **Weighted Metrics**:
  - Precision: 90%
  - Recall: 89%
  - F1-Score: 89%

### Per-Class Performance
- **Food**: Precision: 97%, Recall: 93%, F1-Score: 95%
- **Inside**: Precision: 83%, Recall: 91%, F1-Score: 87%
- **Outside**: Precision: 77%, Recall: 81%, F1-Score: 79%
- **Drink**: Precision: 80%, Recall: 67%, F1-Score: 73%
- **Menu**: Precision: 56%, Recall: 87%, F1-Score: 68%

### Key Observations
- **Best Performance**: "Food" category, benefiting from its dominance in the dataset.
- **Most Challenging Class**: "Menu" due to its limited representation.

---

## Challenges Addressed

### Class Imbalance
- The "menu" category, with only 0.84% of the data, required careful handling to ensure the model didn’t neglect minority classes.
- Adjustments such as data augmentation and regularization helped balance performance.

### Image Validation
- Invalid images were removed during preprocessing, ensuring data quality.

### Diverse Visual Input
- Variations in lighting, angles, and image quality were tackled by using data augmentation techniques (e.g., rotation, flipping).

### Training Optimization
- Early stopping and learning rate scheduling improved training stability and reduced overfitting.

### Evaluation
- Comprehensive metrics such as confusion matrices and per-class performance ensured robust model assessment.

---

## Practical Implications

### User Experience
- Improved photo categorization enables better search and discovery features for Yelp users.

### Business Representation
- Accurate classifications enhance how businesses are showcased, improving customer impressions.

### Scalability
- Automating classification significantly reduces manual effort and operational costs for Yelp.

---

## Future Directions

### Data Augmentation
- Generate synthetic data for minority classes like "menu" to improve model learning.

### Alternative Architectures
- Experiment with models like EfficientNet or Vision Transformers to boost performance.

### Ensemble Learning
- Combine multiple models for better generalization.

### Hyperparameter Optimization
- Use advanced techniques to fine-tune learning rates, batch sizes, and other parameters for improved results.
