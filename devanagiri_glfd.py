import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import math

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DataLoader:
    def __init__(self, data_dir='/kaggle/input/devanagari-handwritten-character-datase/DevanagariHandwrittenCharacterDataset'):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(os.path.join(data_dir, 'Train')))[:10]  # First 10 classes as per paper
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def load_data(self, split=0.7):
        images = []
        labels = []
        
        # Load training data
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, 'Train', class_name)
            class_images = os.listdir(class_dir)[:150]  # 150 images per class as per paper
            
            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(self.class_to_idx[class_name])
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=split, random_state=42
        )
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, len(self.classes))
        y_val = to_categorical(y_val, len(self.classes))
        
        return X_train, X_val, y_train, y_val

class GLFDEnhancer:
    def __init__(self, intensity_factor=1.2, fractional_order=0.7):
        self.kappa = intensity_factor
        self.mu = fractional_order
        
    def create_mask(self):
        window = 3
        mask = np.zeros((window, window))
        
        def M_i(i):
            return (1/math.gamma(-self.mu)) * (
                (math.gamma(i - self.mu + 1)/math.factorial(i + 1)) * ((self.mu**2/8) + self.mu/4) +
                (math.gamma(i - self.mu)/math.factorial(i)) * (1 - self.mu**2/4) +
                (math.gamma(i - self.mu - 1)/math.factorial(i - 1)) * ((self.mu**2/8) - self.mu/4)
            )
        
        center = window // 2
        for i in range(window):
            for j in range(window):
                if i == center and j == center:
                    mask[i,j] = 1 - 8 * M_i(1)
                else:
                    mask[i,j] = M_i(1)
                    
        return mask * self.kappa
    
    def enhance(self, image):
        mask = self.create_mask()
        enhanced = cv2.filter2D(image, -1, mask)
        enhanced = np.clip(enhanced, 0, 255)
        return enhanced.astype(np.uint8)

class DevanagariRecognizer:
    def __init__(self, num_classes=10):
        self.input_shape = (224, 224, 3)
        self.num_classes = num_classes
        self.enhancer = GLFDEnhancer()
        self.model = self._build_model()
        
    def _build_model(self):
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the first 48 layers
        for layer in base_model.layers[:48]:
            layer.trainable = False
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_batch(self, images):
        processed = []
        for image in images:
            # Resize to 256x256 first
            image = cv2.resize(image, (256, 256))
            # Apply random pixel range augmentation ([-35, 35])
            image = np.clip(image + np.random.randint(-35, 36), 0, 255)
            # Apply random scaling (scaleRange [0.8, 1.2])
            scale_factor = np.random.uniform(0.8, 1.2)
            height, width = image.shape
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = cv2.resize(image, new_size)
            # Ensure the image is 224x224
            image = cv2.resize(image, (224, 224))
            # Enhance the image
            enhanced = self.enhancer.enhance(image)
            # Convert to RGB
            enhanced = np.stack([enhanced]*3, axis=-1)
            enhanced = enhanced.astype(np.float32) / 255.0
            processed.append(enhanced)
        return np.array(processed)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=8):
        # Preprocess the data
        X_train_processed = self.preprocess_batch(X_train)
        X_val_processed = self.preprocess_batch(X_val)
        
        # Train the model
        history = self.model.fit(
            X_train_processed,
            y_train,
            validation_data=(X_val_processed, y_val),
            epochs=epochs,
            batch_size=batch_size
        )
        return history
    
    def evaluate_accuracy(self, X_val, y_val):
        # Preprocess validation data
        X_val_processed = self.preprocess_batch(X_val)
        
        # Predict
        zpred = np.argmax(self.model.predict(X_val_processed), axis=-1)
        y_true = np.argmax(y_val, axis=-1)
        
        # Calculate accuracy
        accuracy = np.mean(zpred == y_true)
        return accuracy

# Training script
def main():
    # Load data
    print("Loading data...")
    data_loader = DataLoader()  # Using default path
    X_train, X_val, y_train, y_val = data_loader.load_data()
    print("Data loaded successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {len(data_loader.classes)}")
    print(f"Classes: {data_loader.classes}")
    
    # Initialize recognizer
    print("Initializing model...")
    recognizer = DevanagariRecognizer(num_classes=10)
    
    # Train the model
    print("Starting training...")
    history = recognizer.train(X_train, y_train, X_val, y_val, epochs=10)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate accuracy on validation set
    accuracy = recognizer.evaluate_accuracy(X_val, y_val)
    print(f"Validation accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
