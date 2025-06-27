import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy.special import gamma

# ------------------------- Image Enhancer -------------------------
class ImageEnhancer:
    def __init__(self, intensity_factor=2.0):
        self.intensity_factor = intensity_factor

    def fuzzy_enhance(self, image):
        normalized = image / 255.0
        fuzzy_transformed = normalized**self.intensity_factor / (normalized**self.intensity_factor + (1 - normalized)**self.intensity_factor)
        enhanced = (fuzzy_transformed * 255).astype(np.uint8)
        return enhanced

    def power_law(self, image, alpha=0.6):
        image = image.astype(np.float32)
        enhanced = gamma(1 + alpha) * (image ** (alpha))
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return enhanced

    def enhance(self, image):
        fuzzy_image = self.fuzzy_enhance(image)
        fractional_image = self.power_law(fuzzy_image, alpha=0.5)
        return fractional_image

# ------------------------- Data Loader -------------------------
class DataLoader:
    def __init__(self, data_dir='/kaggle/input/devanagari-handwritten-character-datase/DevanagariHandwrittenCharacterDataset'):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(os.path.join(data_dir, 'Train')))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.enhancer = ImageEnhancer()

    def load_data(self): 
        images, labels = [], []
        
        for class_name in self.classes[:33]:  # 33 classes only
            class_dir = os.path.join(self.data_dir, 'Train', class_name)
            class_images = os.listdir(class_dir)[:300]  # 300 images per class
            
            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img = self.enhancer.enhance(img)
                    images.append(img)
                    labels.append(self.class_to_idx[class_name])
        
        X = np.array(images).reshape(-1, 64, 64, 1)
        y = to_categorical(np.array(labels), num_classes=33)
        return X, y

# ------------------------- Load Data -------------------------
data_loader = DataLoader()
X, y = data_loader.load_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------- Data Augmentation -------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)

# ------------------------- CNN Model -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(33, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

# ------------------------- Training -------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32), 
    epochs=30, 
    validation_data=(X_test, y_test), 
    callbacks=[lr_scheduler]
)

# ------------------------- Evaluation -------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# ------------------------- Predictions -------------------------
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# ------------------------- Classification Report -------------------------
print("Classification Report:")
print(classification_report(y_true, y_pred, digits=4, target_names=data_loader.classes[:33]))

# ------------------------- Confusion Matrix -------------------------
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14,12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=data_loader.classes[:33], yticklabels=data_loader.classes[:33])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ------------------------- ROC Curve -------------------------
fpr, tpr, roc_auc = {}, {}, {}
for i in range(33):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(16, 14))
for i in range(33):
    plt.plot(fpr[i], tpr[i], label=f'{data_loader.classes[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve for Each Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(fontsize='small', loc='lower right')
plt.grid()
plt.show()

# ------------------------- Training/Validation Loss -------------------------
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
