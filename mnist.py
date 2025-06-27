import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import struct
from scipy.special import gamma

# ------------------------- Image Enhancer -------------------------
class ImageEnhancer:
    def __init__(self, intensity_factor=2.0):
        self.intensity_factor = intensity_factor

    def fuzzy_enhance(self, image):
        normalized = image / 255.0
        fuzzy = normalized ** self.intensity_factor / (
            normalized ** self.intensity_factor + (1 - normalized) ** self.intensity_factor + 1e-8
        )
        return (fuzzy * 255).astype(np.uint8)

    def abr_derivative(self, image, alpha=0.6):
        image = image.astype(np.float32)
        enhanced = gamma(1 + alpha) * (image ** alpha)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def enhance(self, image):
        return self.abr_derivative(self.fuzzy_enhance(image), alpha=0.5)

# ------------------------- IDX Reader -------------------------
def read_idx_images(filepath):
    with open(filepath, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

def read_idx_labels(filepath):
    with open(filepath, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# ------------------------- Load Data -------------------------
train_images = read_idx_images('/kaggle/input/mnist-dataset/train-images.idx3-ubyte')
train_labels = read_idx_labels('/kaggle/input/mnist-dataset/train-labels.idx1-ubyte')
test_images = read_idx_images('/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte')
test_labels = read_idx_labels('/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte')

# ------------------------- Optional Enhancement -------------------------
enhancer = ImageEnhancer()

X_train_raw = train_images.astype(np.uint8)
X_test_raw = test_images.astype(np.uint8)

X_train_enhanced = np.array([enhancer.enhance(img) for img in X_train_raw])
X_test_enhanced = np.array([enhancer.enhance(img) for img in X_test_raw])

# Normalize and reshape both versions
def preprocess(X):
    X = X.astype('float32') / 255.
    X = np.expand_dims(X, axis=-1)
    return tf.image.resize(X, [64, 64])

X_train_raw = preprocess(X_train_raw)
X_test_raw = preprocess(X_test_raw)

X_train_enhanced = preprocess(X_train_enhanced)
X_test_enhanced = preprocess(X_test_enhanced)

y_train = to_categorical(train_labels, 10)
y_test = to_categorical(test_labels, 10)

# ------------------------- CNN Architecture -------------------------
def get_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------- Train & Evaluate -------------------------
model_raw = get_model()
model_enhanced = get_model()

history_raw = model_raw.fit(X_train_raw, y_train, validation_data=(X_test_raw, y_test), epochs=10, batch_size=64, verbose=1)
history_enhanced = model_enhanced.fit(X_train_enhanced, y_train, validation_data=(X_test_enhanced, y_test), epochs=10, batch_size=64, verbose=1)

# ------------------------- Accuracy Comparison -------------------------
y_pred_raw = np.argmax(model_raw.predict(X_test_raw), axis=1)
y_pred_enh = np.argmax(model_enhanced.predict(X_test_enhanced), axis=1)

acc_raw = accuracy_score(np.argmax(y_test, axis=1), y_pred_raw)
acc_enh = accuracy_score(np.argmax(y_test, axis=1), y_pred_enh)

print(f"Accuracy WITHOUT enhancement: {acc_raw:.4f}")
print(f"Accuracy WITH enhancement   : {acc_enh:.4f}")

# ------------------------- Plot Comparison -------------------------
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
plt.plot(history_raw.history['val_accuracy'], label='No Enhancement')
plt.plot(history_enhanced.history['val_accuracy'], label='With Enhancement')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history_raw.history['val_loss'], label='No Enhancement')
plt.plot(history_enhanced.history['val_loss'], label='With Enhancement')
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
