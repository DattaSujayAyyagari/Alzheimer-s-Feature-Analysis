# Alzheimer's Disease Classification with XGBoost

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5.0-orange)](https://xgboost.readthedocs.io/en/stable/)

This project demonstrates the use of XGBoost for classifying Alzheimer's disease progression based on various features from a dataset. It includes data preprocessing, model training, evaluation, and hyperparameter tuning.

## Overview

The objective of this project is to build a machine learning model to classify individuals into different stages of Alzheimer's disease: Nondemented, Demented, and Converted. The XGBoost classifier is used for this purpose, leveraging various features from the dataset to make predictions.

## Dataset

The dataset `alzheimer.csv` includes features crucial for diagnosing and tracking Alzheimer's disease progression:
- **Age**: Age of the individual.
- **EDUC**: Education Level.
- **SES**: Socioeconomic Status.
- **MMSE**: Mini-Mental State Examination Score.
- **CDR**: Clinical Dementia Rating.
- **eTIV**: Estimated Total Intracranial Volume.
- **nWBV**: Normalized Whole Brain Volume.
- **ASF**: Atlas Scaling Factor.

## Code Description

### Data Preprocessing

1. **Feature Scaling**: Selected features (`Age`, `EDUC`, `SES`, `MMSE`, `eTIV`, `ASF`) are scaled to a range of 0-1. This is done using min-max scaling to ensure that features are on a comparable scale.
2. **Data Cleaning**: Specific rows are dropped to clean the dataset and ensure consistency.
3. **Encoding**: Categorical variables such as `Group` and `M/F` are encoded into numerical values. This step involves converting categorical text labels into integer values for compatibility with machine learning algorithms.

### Model Training

The XGBoost classifier is trained on the preprocessed dataset:

- **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets.
- **Training**: The model is trained using the training data with XGBoost, configured for binary classification with specific parameters like `objective`, `random_state`, and `eval_metric`.
- **Early Stopping**: Early stopping is used during training to prevent overfitting by stopping training when the model's performance on a validation set no longer improves.

### Evaluation

Model performance is evaluated using the following metrics and techniques:

- **Confusion Matrix**: Provides a summary of prediction results on the test data, showing true positive, true negative, false positive, and false negative predictions.
- **Metrics**: Key performance metrics include accuracy, recall, precision, and F1 score.
- **ROC Curve**: A Receiver Operating Characteristic (ROC) curve is plotted to visualize the trade-off between true positive rate and false positive rate, with the Area Under the Curve (AUC) score indicating overall performance.

### Hyperparameter Tuning

Hyperparameter tuning is performed using `RandomizedSearchCV` to find the optimal set of parameters for the XGBoost model:

- **Parameter Search**: Various hyperparameters such as learning rate, tree depth, and number of estimators are optimized to enhance model performance.
- **Cross-Validation**: The search uses cross-validation to evaluate different combinations of hyperparameters and select the best model.

### Results

- **Best Model**: The best-performing model is identified through hyperparameter tuning and is used to make predictions on the test set.
- **Test Accuracy**: The model's performance is assessed on the test set using metrics such as the weighted F1 score.

### Visualization

Several visualizations are used to illustrate model performance:

- **Confusion Matrix Heatmap**: Shows the distribution of true and predicted classes.
- **ROC Curve**: Plots the true positive rate against the false positive rate, indicating the model's classification capabilities with the calculated AUC.

## Contributing

Contributions to this project are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

You can continue to adjust the README to better fit the context and specifics of your project as needed.
