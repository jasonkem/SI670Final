import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.load()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return True
    except Exception:
        return False

def create_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_generators(train_df, valid_df, PHOTO_DIR):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Ensure batch size is reasonable
    BATCH_SIZE = min(32, len(train_df))
    
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
    
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=PHOTO_DIR,
        x_col='photo_id',
        y_col='label',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, valid_generator, BATCH_SIZE

def main():
    # Paths
    json_path = '/Users/jason/Desktop/Masters/SI 670/Final Project/data/yelp_photos/photos.json'
    PHOTO_DIR = '/Users/jason/Desktop/Masters/SI 670/Final Project/data/yelp_photos/photos'
    
    # Load full data
    print("Loading data...")
    df = pd.read_json(json_path, lines=True)
    df['photo_id'] = df['photo_id'] + '.jpg'
    original_len = len(df)
    
    print(f"\nVerifying {original_len} images for validity...")
    valid_images = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
        img_path = os.path.join(PHOTO_DIR, row['photo_id'])
        if verify_image(img_path):
            valid_images.append(row)
    
    # Create new dataframe with only valid images
    df = pd.DataFrame(valid_images)
    print(f"\nRetained {len(df)} valid images out of {original_len} total images.")
    if len(df) < original_len:
        print(f"{original_len - len(df)} images were invalid and removed.")
    
    # Print class distribution
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Split data
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    
    # Create generators
    train_generator, valid_generator, BATCH_SIZE = create_generators(train_df, valid_df, PHOTO_DIR)
    
    print("\nTraining Configuration:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    
    # Create and compile model
    print("\nCreating model...")
    model = create_model(len(train_generator.class_indices))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('models/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
    ]
    
    try:
        # Increase epochs to 50
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=valid_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        
        # Save the final model as well
        model.save('models/final_model.keras')
        print("\nFinal model saved to 'models/final_model.keras'!")
        
    except Exception as e:
        print(f"\nTraining error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
