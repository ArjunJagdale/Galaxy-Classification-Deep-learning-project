# Classifying Galaxies Using Convolutional Neural Networks

## Overview
Telescopes worldwide capture millions of images of celestial objects, including galaxies, stars, and planets. These datasets are invaluable to astronomers but require annotation to be useful for scientific analysis. This project leverages deep learning to classify galaxies into four categories based on their unique characteristics, using data from the Galaxy Zoo initiative.

The goal is to build and train a Convolutional Neural Network (CNN) to classify galaxies into the following classes:
1. **Normal Galaxies**: No distinguishing features.
2. **Ringed Galaxies**: Galaxies with star rings.
3. **Galactic Mergers**: Two galaxies in the process of merging.
4. **Irregular Celestial Bodies**: Objects with unusual shapes or configurations.

## Steps Performed

### 1. Data Preparation
- The dataset was loaded using a custom `load_galaxy_data()` function.
- The dimensions of the data and labels were inspected using `.shape`.
- The dataset was split into training and validation subsets using `train_test_split` with 80% for training and 20% for validation, ensuring stratification for balanced class distribution.

### 2. Data Preprocessing
- An `ImageDataGenerator` was used to normalize pixel values by rescaling them to the range [0, 1].
- Two NumpyArrayIterators were created using `.flow()`, one for training data and the other for validation data, with a batch size of 5.

### 3. Model Architecture
The CNN model was designed using the following architecture:
- **Input Layer**: Shape `(128, 128, 3)` to match image dimensions.
- **Two Convolutional Layers**:
  - 8 filters, each 3x3, with strides of 2 and ReLU activation.
- **Two MaxPooling Layers**:
  - Pool size `(2, 2)` with strides of 2.
- **Flatten Layer**: To prepare the data for dense layers.
- **Hidden Dense Layer**: 16 units with ReLU activation.
- **Output Dense Layer**: 4 units with softmax activation for multi-class classification.

### 4. Compilation
The model was compiled using:
- Optimizer: `Adam` with a learning rate of 0.001.
- Loss Function: `CategoricalCrossentropy` for one-hot encoded labels.
- Metrics: `CategoricalAccuracy` and `AUC`.

### 5. Training
- The model was trained for 8 epochs with:
  - Training iterator as input.
  - Steps per epoch calculated as `len(x_train) / batch_size`.
  - Validation data and steps provided.

### 6. Results
After training, the model achieved:
- **Accuracy**: ~60-70%
- **AUC**: ~80-90%

### 7. Insights
- The model accurately identified the correct galaxy class over 60% of the time, significantly surpassing the random baseline accuracy of 25%.
- AUC values above 80% indicated strong model performance in distinguishing between true and false classes.

### 8. Visualization
Feature maps of convolutional layers were visualized using a `visualize_activations()` function to understand how the network processed the input images.

## Conclusion
This project successfully trained a CNN to classify galaxies, demonstrating the power of deep learning in astronomical image analysis. Further improvements could include:
- Optimizing hyperparameters (e.g., learning rate, number of filters).
- Experimenting with additional convolutional layers or larger dense layers.
- Preventing overfitting with regularization techniques like dropout.

## Code Snippet
```python
# Import libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

# Load and preprocess data
input_data, labels = load_galaxy_data()
x_train, x_valid, y_train, y_valid = train_test_split(
    input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

data_generator = ImageDataGenerator(rescale=1./255)
training_iterator = data_generator.flow(x_train, y_train, batch_size=5)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

# Build and compile the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])

# Train the model
model.fit(training_iterator, steps_per_epoch=len(x_train)/5, epochs=8,
          validation_data=validation_iterator, validation_steps=len(x_valid)/5)

# Visualize activations
from visualize import visualize_activations
visualize_activations(model, validation_iterator)
