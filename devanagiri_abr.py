# Image enhancemet + Simple CNN
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from scipy.special import gamma, factorial
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from scipy.signal import convolve2d
from mpmath import mp

mp.dps = 50  # High precision for Mittag-Leffler function

class ABREnhancer:
    def __init__(self, alpha=0.4, beta=0.8, gamma_factor=0.7, intensity_factor=2.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma_factor = gamma_factor
        self.intensity_factor = intensity_factor
    
    def fuzzy_enhance(self, image):
        normalized = image / 255.0
        fuzzy_transformed = normalized**self.intensity_factor / (normalized**self.intensity_factor + (1 - normalized)**self.intensity_factor)
        enhanced = (fuzzy_transformed * 255).astype(np.uint8)
        return enhanced
    
    def mittag_leffler_approx(self, alpha, x, terms=3):
        """Approximate Mittag-Leffler function using power series"""
        result = 0
        for k in range(terms):
            result += (x ** k) / gamma(alpha * k + 1)
        return result
    
    def fractional_derivative(self, image, beta):
        """Compute fractional derivative using simplified approach"""
        # Convert to float for calculations
        img_float = image.astype(np.float32)
        
        # Apply horizontal and vertical differences
        dx = np.diff(img_float, axis=1, prepend=img_float[:, :1])
        dy = np.diff(img_float, axis=0, prepend=img_float[:1, :])
        
        # Combine directional derivatives with fractional power
        magnitude = np.sqrt(dx**2 + dy**2)
        fractional_mag = magnitude**beta
        
        return fractional_mag
    
    def abr_fractional_enhancement(self, image):
        """Implements simplified ABR enhancement"""
        # Normalize image
        img_norm = image.astype(np.float32) / 255.0
        
        # Calculate ABR parameter
        B_alpha = (self.alpha + gamma(self.alpha + 1) * (1 - self.alpha)) / gamma(self.alpha + 1)
        
        # Apply fractional derivative
        frac_derivative = self.fractional_derivative(img_norm, self.beta)
        
        # Apply ABR formula (simplified)
        enhanced = B_alpha * (img_norm + (1 / (1 - self.alpha)) * frac_derivative)
        
        # Return normalized result
        return np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    
    def apply_abr_mask(self, image):
        """Apply ABR fractal-fractional mask"""
        mask = np.array([
            [self.gamma_factor, self.gamma_factor, self.gamma_factor],
            [self.gamma_factor, 1 - 8 * self.gamma_factor, self.gamma_factor],
            [self.gamma_factor, self.gamma_factor, self.gamma_factor]
        ])
        enhanced = convolve2d(image, mode='same', boundary='symm', in2=mask)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def enhance(self, image):
        """Full enhancement pipeline"""
        # Step 1: Apply fuzzy enhancement
        fuzzy_image = self.fuzzy_enhance(image)
        
        # Step 2: Apply ABR fractional enhancement
        fractional_enhanced = self.abr_fractional_enhancement(fuzzy_image)
        
        # Step 3: Apply ABR mask for final enhancement
        final_enhanced = self.apply_abr_mask(fractional_enhanced)
        
        return final_enhanced

    
# DataLoader Class with Enhancement
class DataLoader:
    def __init__(self, data_dir='/kaggle/input/devanagari-handwritten-character-datase/DevanagariHandwrittenCharacterDataset'):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(os.path.join(data_dir, 'Train')))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # self.enhancer = ImageEnhancer()
        self.enhancer = ABREnhancer()# Instantiate the ImageEnhancer

    def load_data(self):
        images, labels = [], []
        
        for class_name in self.classes[:33]:  # Ensure we only load 33 classes
            class_dir = os.path.join(self.data_dir, 'Train', class_name)
            class_images = os.listdir(class_dir)[:300]  # 300 images per class
            
            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Normalize size
                    img = self.enhancer.enhance(img)  # Apply enhancement
                    images.append(img)
                    labels.append(self.class_to_idx[class_name])
        
        X = np.array(images).reshape(-1, 64, 64, 1)  # Reshape to (64,64,1)
        y = to_categorical(np.array(labels), num_classes=33)  # One-hot encoding

        return X, y

# Load dataset with enhancement
data_loader = DataLoader()
X, y = data_loader.load_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# CNN Model with Dropout for Regularization
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Added Dropout to prevent overfitting
    Dense(33, activation='softmax')  # Output layer for 33 classes
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Learning Rate Scheduling (Reduce LR if validation loss stagnates)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

datagen = ImageDataGenerator(
    rotation_range=30,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Zoom transformations
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill missing pixels after transformations
)

# Train model with augmented data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32), 
    epochs=50, 
    validation_data=(X_test, y_test), 
    callbacks=[lr_scheduler]  # Apply learning rate scheduler
)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')