Alzheimer's Disease Classification with XGBoost
This project demonstrates the use of machine learning for classifying Alzheimer's disease based on various features. The model employs the XGBoost classifier to distinguish between different stages of Alzheimer's disease.

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
Overview
The goal of this project is to build a machine learning model to classify individuals into different groups based on Alzheimer's disease progression. The groups are:

Nondemented
Demented
Converted
Dataset
The dataset used for this project is assumed to be alzheimer.csv. The key features included in the dataset are:

Age
EDUC: Education Level
SES: Socioeconomic Status
MMSE: Mini-Mental State Examination Score
CDR: Clinical Dementia Rating
eTIV: Estimated Total Intracranial Volume
nWBV: Normalized Whole Brain Volume
ASF: Atlas Scaling Factor
Setup
Prerequisites
Python 3.x
Required libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, scipy
Installation
Install the necessary Python packages using pip:

bash
Copy code
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
Data
Place your dataset (alzheimer.csv) in the working directory.

Data Preprocessing
The following preprocessing steps are performed:

Scaling: Features Age, EDUC, SES, MMSE, eTIV, and ASF are scaled to a range of 0-1.
Data Cleaning: Some rows are dropped to prepare the dataset.
Encoding: The target variable Group and feature M/F are encoded into numerical values.
Model Training
The XGBoost classifier is trained using the processed dataset. The training and testing datasets are split with an 80-20 ratio.

python
Copy code
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

xgb_model = XGBClassifier(objective="binary:logistic", random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(train_x, train_y, verbose=0, early_stopping_rounds=5, eval_set=[(test_x, test_y)])
Evaluation
Performance metrics for the model include:

Confusion Matrix
Accuracy
Recall
Precision
Classification Report
ROC Curve
python
Copy code
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
Hyperparameter Tuning
Hyperparameters for the XGBoost model are optimized using RandomizedSearchCV.

python
Copy code
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

parameters = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), 
    "max_depth": randint(2, 6), 
    "n_estimators": randint(100, 150), 
    "subsample": uniform(0.6, 0.4)
}

search = RandomizedSearchCV(xgb_model, param_distributions=parameters, random_state=42, n_iter=200, cv=3, verbose=0, n_jobs=1, return_train_score=True, scoring='f1_weighted')
search.fit(x, y, verbose=0)
Results
The best model is evaluated on the test set:

python
Copy code
best_model = search.best_estimator_
test_accuracy = best_model.score(test_x, test_y)
print(f'Test set weighted f1 score of the best model: {test_accuracy:.3f}')
Visualization
The project provides visualizations such as the confusion matrix and ROC curve to understand model performance.

Confusion Matrix
python
Copy code
import seaborn as sb

conf_matrix = confusion_matrix(test_y, predict_y)
sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar_kws={'shrink': .3}, linewidths=.1)
plt.show()
ROC Curve
python
Copy code
false_positive_rate, true_positive_rate, _ = roc_curve(test_y, predict_y)
auc = roc_auc_score(predict_y, test_y)
plt.plot(false_positive_rate, true_positive_rate, label=f"AUC={auc:.3f}")
plt.title('ROC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
Contributing
Contributions are welcome! Please open an issue or submit a pull request with improvements.

License
This project is licensed under the MIT License."# Alzheimer-s-Feature-Analysis" 
