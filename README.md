# 🔥 Wildfire Detection using Deep Learning

This project leverages deep learning techniques to predict and detect wildfires based on environmental and sensor data. By training a neural network on wildfire-related features, we aim to improve accuracy and reliability in early wildfire detection.

---

## 📁 Dataset

**Source:** [Kaggle - The Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset)

This dataset contains sensor and environmental readings including temperature, humidity, wind, and more, labeled with whether a wildfire occurred.

---

## 🎯 Project Goals

- Use a deep learning model (Dense Neural Network) for classification
- Enhance detection accuracy compared to traditional ML models
- Preprocess and scale the data for optimal neural network performance
- Evaluate model with key metrics (accuracy, precision, recall, etc.)

---

## 🧰 Tools & Libraries

- Python 3.x 🐍
- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (optional for visualization)

---

## 📊 Model Architecture

> A fully connected Deep Neural Network (DNN):

```text
Input Layer (features)
↓
Dense(128, activation='relu')
↓
Dropout(0.3)
↓
Dense(64, activation='relu')
↓
Dropout(0.3)
↓
Dense(32, activation='relu')
↓
Output Layer (1 neuron, activation='sigmoid')
