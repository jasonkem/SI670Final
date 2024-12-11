import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tqdm import tqdm

def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def evaluate_saved_model():
    # Load the saved model
    print("Loading saved model...")
    model = tf.keras.models.load_model('models/best_model.keras')
    
    # Paths
    json_path = '/Users/jason/Desktop/Masters/SI 670/Final Project/data/yelp_photos/photos.json'
    PHOTO_DIR = '/Users/jason/Desktop/Masters/SI 670/Final Project/data/yelp_photos/photos'
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_json(json_path, lines=True)
    df['photo_id'] = df['photo_id'] + '.jpg'
    
    # Verify images first
    print("\nVerifying images...")
    valid_images = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(PHOTO_DIR, row['photo_id'])
        if verify_image(img_path):
            valid_images.append(row)
    
    # Create new dataframe with only valid images
    df = pd.DataFrame(valid_images)
    
    # Use same train/test split as original
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Create test generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=PHOTO_DIR,
        x_col='photo_id',
        y_col='label',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate and print per-class metrics
    print("\nPer-class Metrics:")
    for class_name, class_idx in test_generator.class_indices.items():
        true_mask = (y_true == class_idx)
        pred_mask = (y_pred == class_idx)
        
        true_count = np.sum(true_mask)
        pred_count = np.sum(pred_mask)
        correct_count = np.sum((y_pred == class_idx) & (y_true == class_idx))
        
        if true_count > 0:
            accuracy = correct_count / true_count
            print(f"\n{class_name}:")
            print(f"  True samples: {true_count}")
            print(f"  Predicted samples: {pred_count}")
            print(f"  Correct predictions: {correct_count}")
            print(f"  Accuracy: {accuracy:.4f}")
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # Plot and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/quick_confusion_matrix.png')
    plt.close()
    
    print("\nResults saved to outputs/quick_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_saved_model() 