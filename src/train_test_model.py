import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_generators(train_df, valid_df, test_df, PHOTO_DIR):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    BATCH_SIZE = 32
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=PHOTO_DIR,
        x_col='photo_id',
        y_col='label',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    valid_generator = valid_test_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=PHOTO_DIR,
        x_col='photo_id',
        y_col='label',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = valid_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=PHOTO_DIR,
        x_col='photo_id',
        y_col='label',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, test_generator, BATCH_SIZE

def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

def evaluate_model(model, generator, num_samples):
    # Convert steps to integer using int()
    steps = int(np.ceil(num_samples/generator.batch_size))
    
    # Get predictions
    predictions = model.predict(generator, steps=steps)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = generator.classes[:len(predicted_classes)]
    
    # Calculate MSE
    mse = mean_squared_error(
        tf.keras.utils.to_categorical(true_classes, num_classes=len(generator.class_indices)),
        predictions
    )
    
    return true_classes, predicted_classes, mse

def main():
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created outputs directory")
    
    # Paths
    json_path = '/Users/jason/Desktop/Masters/SI 670/Final Project/data/yelp_photos/photos.json'
    PHOTO_DIR = '/Users/jason/Desktop/Masters/SI 670/Final Project/data/yelp_photos/photos'
    
    # Load full data
    print("Loading data...")
    df = pd.read_json(json_path, lines=True)
    df['photo_id'] = df['photo_id'] + '.jpg'
    original_len = len(df)
    
    print(f"\nProcessing all {original_len} images...")
    
    print("\nVerifying images...")
    valid_images = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        img_path = os.path.join(PHOTO_DIR, row['photo_id'])
        if verify_image(img_path):
            valid_images.append(row)
    
    # Create new dataframe with only valid images
    df = pd.DataFrame(valid_images)
    print(f"\nRetained {len(df)} valid images out of {original_len} total images.")
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['label'].value_counts())
    print("\nPercentage distribution:")
    print(df['label'].value_counts(normalize=True).mul(100).round(2), "%")
    
    # First split: separate test set (20% of data)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.2,
        random_state=42,
        stratify=df['label']  # This ensures proportional split
    )
    
    # Second split: separate training and validation (80/20 split of remaining data)
    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df['label']  # This ensures proportional split
    )
    
    print("\nData split sizes:")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"Validation samples: {len(valid_df)} ({len(valid_df)/len(df):.1%})")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    # Print class distribution in each split
    print("\nClass distribution in splits:")
    print("\nTraining set:")
    print(train_df['label'].value_counts())
    print("\nValidation set:")
    print(valid_df['label'].value_counts())
    print("\nTest set:")
    print(test_df['label'].value_counts())
    
    # Create generators
    train_generator, valid_generator, test_generator, BATCH_SIZE = create_generators(
        train_df, valid_df, test_df, PHOTO_DIR
    )
    
    # Create and compile model
    print("\nCreating model...")
    model = create_model(len(train_generator.class_indices))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model_with_test.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    print("\nStarting training...")
    try:
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=valid_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = model.evaluate(
            test_generator,
            steps=len(test_df) // BATCH_SIZE,
            verbose=1
        )
        print(f"\nTest set accuracy: {test_results[1]:.4f}")
        
        # Get predictions and calculate MSE
        true_classes, predicted_classes, mse = evaluate_model(
            model, test_generator, len(test_df)
        )
        
        print(f"\nMean Squared Error: {mse:.4f}")
        
        # Plot confusion matrix
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(
            true_classes,
            predicted_classes,
            classes=list(test_generator.class_indices.keys())
        )
        print("Confusion matrix saved as 'outputs/confusion_matrix.png'")
        
        # Calculate per-class accuracy
        print("\nPer-class accuracy:")
        for class_name, class_idx in test_generator.class_indices.items():
            class_mask = (true_classes == class_idx)
            if np.sum(class_mask) > 0:  # Check if we have any samples for this class
                class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
                print(f"{class_name}: {class_accuracy:.4f}")
            else:
                print(f"{class_name}: No samples in test set")
        
        # Add detailed prediction counts
        print("\nPrediction counts per class:")
        for class_name, class_idx in test_generator.class_indices.items():
            true_count = np.sum(true_classes == class_idx)
            pred_count = np.sum(predicted_classes == class_idx)
            print(f"{class_name}:")
            print(f"  True samples: {true_count}")
            print(f"  Predicted samples: {pred_count}")
        
        # Save final model
        model.save('models/final_model.keras')
        print("\nFinal model saved to 'models/final_model.keras'!")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png')
        plt.close()
        print("Training history plot saved as 'outputs/training_history.png'")
        
    except Exception as e:
        print(f"\nTraining error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 