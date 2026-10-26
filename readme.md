# Sensor-Based Human Activity Classification

This project involves developing a deep learning model to classify human activity using data collected from a Passive Infra-Red (PIR) sensor. The dataset records human occupancy in both residential and office environments over a 4-second interval. It includes both tabular features (ambient temperature, date, time, and activity labels) and time series data (55 analog PIR sensor readings).

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Libraries and Dependencies](#libraries-and-dependencies)
4. [Installation and Execution](#installation-and-execution)
5. [Code Structure and Workflow](#code-structure-and-workflow)
6. [Model Training and Validation](#model-training-and-validation)
7. [Checkpointing and Evaluation Function](#checkpointing-and-evaluation-function)
8. [Usage Guidelines](#usage-guidelines)
9. [Conclusion](#conclusion)

## Project Overview

This project focuses on building and evaluating robust classification models for detecting human activities using sensor data. The major steps include:

- **Data Preprocessing:**  
  Handling missing values, computing statistical features from the PIR sensor time series, scaling features, and encoding activity labels.
- **Data Split:**  
  Dividing the dataset into 70% training and 30% testing sets.
- **Cross-Validation:**  
  Applying 5-fold cross-validation on the training data.
- **Handling Imbalance:**  
  Using SMOTE oversampling to address class imbalance.
- **Model Training:**  
  Training two deep learning models, a Transformer-based classifier and an LSTM-based model, with early stopping based on validation accuracy.
- **Retraining:**  
  Re-training the best performing model on the full training set and saving the checkpoint.
- **Evaluation:**  
  Evaluating the final model on the unseen 30% test dataset using a dedicated evaluation function.

## Dataset Description

**Dataset:** PIRvision FoG Presence Detection  
**Source:** UCI PIRvision Dataset

### Features

- **Tabular Features:**
  - Date
  - Time
  - Label (activity)
  - Temperature

- **Time Series Features:**
  - 55 consecutive PIR sensor readings over a 4-second window

*Preprocessing Steps:*
- Compute additional statistical features (mean, standard deviation, max, min, range, sum, median, variance) from the 55 PIR sensor readings.
- Drop irrelevant columns (e.g., Date and Time).

## Libraries and Dependencies

The project utilizes several Python libraries:

- **Data Manipulation:** `numpy`, `pandas`
- **Deep Learning:** `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
- **Machine Learning:** `sklearn` (for preprocessing, cross-validation, evaluation metrics)
- **Imbalanced Data Handling:** `imbalanced-learn` (SMOTE)
- **Visualization:** `matplotlib`, `seaborn`
- **OS/File Handling:** `os`

## Installation and Execution

### Required Packages

Install only the necessary packages:

```bash
pip install numpy pandas torch scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### Dataset Setup

Place the dataset file at:
```
Dataset/pirvision_fog_presence_detection/pirvision_office_dataset1.csv
```

### Running the Project

Start Jupyter Notebook and run the notebook:
```bash
jupyter notebook
```

Open the notebook file and execute all cells sequentially.

## Code Structure and Workflow

### Preprocessing:
- **Function:** `preprocess_dataframe(df)`
- Processes the dataframe by filling missing values, computing PIR statistical features, scaling features, and encoding activity labels.

### Model Architectures:

- **Transformer Classifier:**
  - Contains an embedding layer, positional encoding, transformer encoder layers, adaptive average pooling, and a classifier head.
  - Xavier initialization is used for weight initialization.

- **LSTM Model:**
  - Designed to capture the sequential dependencies in the time series data.

### Training and Cross-Validation:

- **Cross-Validation:**
  - 5-fold stratified cross-validation on the 70% training dataset.
  
- **Oversampling:**
  - SMOTE is applied to address class imbalance.
  
- **Training Function:**
  - `train_model()` handles model training with early stopping based on validation accuracy.
  
- **Metrics:**
  - Accuracy, Macro F1-score, and confusion matrices are computed and visualized.

### Retraining and Checkpointing:
- After cross-validation, the best performing model is retrained on the full training data.
- The final model is saved as a checkpoint for later evaluation.

### Evaluation Function:
- **Function:** `evaluate_model(datafile_location, checkpoint)`
- Loads the dataset, preprocesses it, loads the saved checkpoint, and evaluates the model on the 30% test dataset.

Default file path in the code:
```python
test_file_location = "Dataset/pirvision_fog_presence_detection/pirvision_office_dataset1.csv"
```

### Visualizations:
- Displays training vs. validation loss and accuracy curves.
- Shows confusion matrices for each cross-validation fold and the final test evaluation.

## Model Training and Validation

- **Data Split:**
  - 70% training / 30% testing

- **Cross-Validation:**
  - 5-fold stratified cross-validation on the training dataset.

- **Metrics:**
  - Accuracy and Macro F1-score for each fold.

- **Model Optimization:**
  - Early stopping based on validation accuracy is applied to prevent overfitting.

- **Final Model:**
  - The best performing model is retrained on the complete training set, and its weights are saved as a checkpoint.

## Checkpointing and Evaluation Function

- **Model Checkpoint:**
  - The final retrained model is saved at:
  ```python
  final_checkpoint_path = "Dataset/pirvision_fog_presence_detection/best_final_model_lstm.pt"
  ```
  - The checkpoint is created in the same directory as the dataset (`Dataset/pirvision_fog_presence_detection/`) with the filename `best_final_model_lstm.pt`
  **Note:** This is for LSTM code and not Transformer model.
  - This saved model contains the weights of the best performing LSTM model after training on the full training dataset

- **Evaluation Function:**
  - The `evaluate_model` function loads the checkpoint and computes the accuracy on the test dataset, providing flexibility to change the dataset file path if needed.

## Usage Guidelines

- **Running the Notebook:**
  - Execute each cell sequentially in the provided Jupyter Notebook.

- **Verify Data Paths:**
  - Ensure that the dataset file path is correct. Modify `test_file_location` if a different test file is used.

- **Visual Analysis:**
  - Review the generated plots (loss and accuracy curves, confusion matrices) to understand the model's performance.

- **Reproducibility:**
  - A fixed random seed (e.g., `random_state=42`) is used to ensure reproducibility. Maintain this standard if any changes are made.

## Conclusion

In evaluating two different architectures on sequential PIR sensor data:
* **Transformer Model:** Achieved an accuracy of **0.9747**.
* **LSTM Model:** Achieved an accuracy of **0.9856**.

The LSTM model outperforms the Transformer model for this task, indicating its superior ability to capture temporal dependencies in the data. This emphasizes the importance of selecting a model architecture that is well-suited to the characteristics of the input data.