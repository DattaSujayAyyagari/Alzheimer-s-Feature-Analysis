Alzheimer's Disease Classification with XGBoost
This project demonstrates the use of machine learning to classify the stages of Alzheimer's disease. We use the XGBoost classifier to predict and distinguish between different stages of the disease.

Table of Contents
Overview
Dataset
Setup
Data Preprocessing
Model Training
Evaluation
Hyperparameter Tuning
Results
Visualization
Contributing
License
Overview
The objective of this project is to build a machine learning model to classify individuals into different Alzheimer's disease progression groups based on various features. The classification groups include:

Nondemented
Demented
Converted
Dataset
The dataset, alzheimer.csv, contains features that are critical for Alzheimer's disease diagnosis and progression tracking:

Age: Age of the individual.
EDUC: Education Level.
SES: Socioeconomic Status.
MMSE: Mini-Mental State Examination Score.
CDR: Clinical Dementia Rating.
eTIV: Estimated Total Intracranial Volume.
nWBV: Normalized Whole Brain Volume.
ASF: Atlas Scaling Factor.
The dataset is preprocessed to include these features along with encoded categorical variables such as gender and disease group.

Setup
Prerequisites
Ensure you have the following installed:

Python 3.x
Python libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, scipy
Installation
Install the required Python packages using the following command:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
Data
Place your alzheimer.csv file in the working directory.

Data Preprocessing
Scaling: Features like Age, EDUC, SES, MMSE, eTIV, and ASF are scaled to a range of 0-1.
Data Cleaning: Specific rows are removed to ensure a clean dataset.
Encoding: Target variable Group and feature M/F are encoded into numerical values to facilitate model training.
Model Training
The processed data is split into training and testing sets. The XGBoost classifier is trained on the training dataset. The training process includes tuning the model using parameters such as learning rate, maximum depth, and the number of estimators.

Evaluation
The model's performance is evaluated using various metrics:

Confusion Matrix: Visualizes the accuracy of the predictions.
Accuracy: Overall correctness of the model's predictions.
Recall: Ability of the model to find all the relevant cases.
Precision: Accuracy of the positive predictions.
Classification Report: Provides a detailed report including precision, recall, and F1 score.
ROC Curve: Illustrates the true positive rate against the false positive rate.
Hyperparameter Tuning
Hyperparameters for the XGBoost model are optimized using RandomizedSearchCV. This involves searching over a range of values for parameters like learning rate, depth of trees, and number of estimators to find the best combination that improves model performance.

Results
The best model is identified through hyperparameter tuning and evaluated on the test set. The evaluation includes metrics such as weighted F1 score, accuracy, and the area under the ROC curve (AUC).

Visualization
Visualizations include:

Confusion Matrix: Highlights the performance of the model in terms of correctly and incorrectly classified instances.
ROC Curve: Shows the trade-off between true positive rate and false positive rate.
Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.
