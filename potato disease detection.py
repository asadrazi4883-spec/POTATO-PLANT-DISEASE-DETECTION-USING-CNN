from google.colab import drive 
drive.mount('/content/drive')

# INSTALL & IMPORT 

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import random

#STEP 3: Load Dataset 
data_dir = "/content/drive/MyDrive/Potato_plant_disease/dataset" 
data = tf.keras.preprocessing.image_dataset_from_directory( 
    data_dir, 
    image_size=(224, 224), 
    batch_size=32, 
    shuffle=True
)
class_names = data.class_names 
print("Classes:", class_names)

#STEP 4: Use CORRECT Preprocessing (VGG16 FIX) 

from tensorflow.keras.applications.vgg16 import preprocess_input 
data = data.map(lambda x, y: (preprocess_input(x), y))

#STEP 5: Train-Test Split 

train_size = int(0.8 * len(data)) 
train = data.take(train_size)
test = data.skip(train_size)
                  
#STEP 6: Data Augmentation (IMPROVES GENERALIZATION) 
data_augmentation = keras.Sequential([ 
    layers.RandomFlip("horizontal"), 
    layers.RandomRotation(0.1),
])
# STEP 7: Load VGG16 
base_model = tf.keras.applications.VGG16( 
    input_shape=(224, 224, 3), 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = True 
for layer in base_model.layers[:-4]: 
    layer.trainable = False
#STEP 8: Build Model 
model = keras.Sequential([ 
    data_augmentation, 
    base_model, layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dropout(0.5), 
    layers.Dense(len(class_names), activation='softmax')
])

#STEP 9: Compile Model 
model.compile( 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
#STEP 10: Early Stopping 
from tensorflow.keras.callbacks import EarlyStopping 
early_stop = EarlyStopping( 
    monitor='val_accuracy', 
    patience=3, 
    restore_best_weights=True
)
# STEP 11: Train Model 
history = model.fit( 
    train, 
    validation_data=test, 
    epochs=15, 
    #callbacks=[early_stop]
)
# STEP 12: Evaluate 
model.evaluate(test)

# STEP 13: Upload Image for Prediction

from google.colab import files 
uploaded = files.upload() 
img_path = list(uploaded.keys())[0]
 
 # STEP 14: Predict 
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224)) 
plt.imshow(img) 
plt.axis('off')
 
 
img_array = keras.preprocessing.image.img_to_array(img) 
img_array = np.expand_dims(img_array, axis=0) 
img_array = preprocess_input(img_array) 
prediction = model.predict(img_array) 
score = prediction[0]

# STEP 15: Output

print("\nPrediction Probabilities:") 
for i in range(len(class_names)): 
     print(f"{class_names[i]}: {score[i]*100:.2f}%")
predicted_class = class_names[np.argmax(score)].replace("Potato___", "") 
confidence = np.max(score) * 100 
print(f"\nFinal Prediction: {predicted_class}") 
print(f"Confidence: {confidence:.2f}%") 
loss, accuracy = model.evaluate(test) 
print(f"\n Model Accuracy: {accuracy*100:.2f}%") 
print("\nPrediction Probabilities:") 
for i in range(len(class_names)): 
    print(f"{class_names[i]}: {score[i]*100:.2f}%")
# BAR GRAPH FOR PROBABILITIES 
plt.figure(figsize=(6,4)) 
plt.bar(class_names, score) 
plt.xlabel("Classes")
plt.ylabel("Probability") 
plt.title("Prediction Confidence") 
plt.xticks(rotation=30) 
plt.show()

# Final Prediction 
predicted_class = class_names[np.argmax(score)].replace("Potato___", "") 
confidence = np.max(score) * 100 
print(f"\nFinal Prediction: {predicted_class}") 
print(f"Confidence: {confidence:.2f}%")


# Accuracy Graph 
plt.figure(figsize=(8,5)) 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.title('Accuracy vs Epochs') 
plt.legend() 
plt.show()


# Loss Graph
plt.figure(figsize=(8,5)) 
plt.plot(history.history['loss'], label='Train Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.title('Loss vs Epochs') 
plt.legend() 
plt.show()

