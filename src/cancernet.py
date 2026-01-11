import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import numpy as np

# Dataset path
DATA_DIR = r"C:\Users\dell\OneDrive\Desktop\Breast_Cancer_Classification\data"

# Function to remove unreadable images
def remove_corrupted_images(folder):
    count = 0
    for subdir, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError):
                os.remove(file_path)
                count += 1
    print(f"Removed {count} corrupted images from {folder}")

# Clean both folders
remove_corrupted_images(os.path.join(DATA_DIR, "benign"))
remove_corrupted_images(os.path.join(DATA_DIR, "malignant"))

# Image generator with validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Limit samples
MAX_TRAIN = 20000
MAX_VAL = 5000

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(50,50),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# ===== CLASS WEIGHTS (ADD THIS BLOCK) =====
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
# ========================================

train_gen.samples = min(train_gen.samples, MAX_TRAIN)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(50,50),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
val_gen.samples = min(val_gen.samples, MAX_VAL)

# CancerNet CNN
base_model = MobileNetV2(
    input_shape=(50, 50, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = True

# Freeze first 100 layers, fine-tune the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model (5 epochs)
history = model.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    class_weight=class_weights   
)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/cancernet_model.h5")

# Evaluate
val_gen.reset()
preds = model.predict(val_gen)
pred_labels = (preds > 0.5).astype(int)
true_labels = val_gen.classes[:len(pred_labels)]

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", classification_report(true_labels, pred_labels))

# Save confusion matrix
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("results/confusion_matrix.png")
plt.close()

# Save accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Accuracy Plot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("results/accuracy_plot.png")
plt.close()

print("✅ Training complete. Model, confusion matrix, and accuracy plot saved.")
