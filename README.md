# ANN Classification – Breast Cancer Prediction

## 🧠 Overview
This project implements an *Artificial Neural Network (ANN) for classification* using TensorFlow & Keras.  
The dataset is based on *breast cancer prediction (binary classification), where the model predicts whether a tumor is **benign or malignant*.  
The workflow includes *EDA, preprocessing, ANN creation, overfitting handling, training, and evaluation*.  

## 📖 About ANN for Classification
An *Artificial Neural Network (ANN)* can effectively solve *classification problems* by learning non-linear relationships in the data.  
For binary classification, the *sigmoid activation function* is used in the output layer to produce probabilities between *0 and 1*.  

## 🛠 Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- Scikit-learn (preprocessing, classification report, confusion matrix)  

## ⚙️ Workflow
1. *Exploratory Data Analysis (EDA)*  
   - Visualized class balance and feature correlations.  

2. *Preprocessing*  
   - Train-test split.  
   - Scaled features using StandardScaler.  

3. *Model Creation*  
   - Built ANN using *Sequential API* with Dense layers.  
   - Added *Dropout layers* to prevent overfitting.  
   - Optimizer: *Adam*  
   - Loss: *Binary Crossentropy*  
   - Output Layer: *Sigmoid activation* for binary classification.  

4. *Training*  
   - *Initial training* with many epochs → resulted in overfitting.  
   - Applied *EarlyStopping (Keras Callbacks)* to stop training when validation loss stopped improving.  
   - Retrained model with *Dropout layers* for better generalization.  

5. *Evaluation*  
   - Evaluated with:  
     - *Classification Report (Precision, Recall, F1-score)*  
     - *Confusion Matrix*  

## 📂 Code Included
- EDA & visualization  
- Train-test splitting & scaling  
- ANN model creation with Dense, Activation, and Dropout  
- Model compilation with Adam optimizer & Binary Crossentropy  
- Training with and without EarlyStopping  
- Evaluation with Classification Report & Confusion Matrix
