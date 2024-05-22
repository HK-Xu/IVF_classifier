from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score 
from sklearn.model_selection import train_test_split, LeaveOneOut, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

freq_thr = 0.25 #(0-1), frequency threshold for features selected in all LOO iterations
top_percent = 10 #(0-100), threshold for RF importamce
rs = 35
lw = 0.3

## Data input
data = pd.read_csv('Features_Quant_processed.tsv',sep='\t',index_col=0)
X = data.iloc[:,:]
X_copy = X
mean_values = X_copy[X_copy != 0].mean()
X = X_copy.replace(0, mean_values)

## Define group labels
train_Y = [str(i).split('_')[2] for i in data.index.tolist()]
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(train_Y)
Y = pd.DataFrame(Y,index=data.index.tolist())  # LD:1,H:0

## models
# Paremeters for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
# GridSearchCV 10-fold 
clf = GridSearchCV(svm.SVC(decision_function_shape='ovr',probability=True,kernel='linear',random_state=82), param_grid, refit=True, verbose=2, cv=10)

now = datetime.now()
print("Current time: ", now.strftime("%Y-%m-%d %H:%M:%S")) 
    
random.seed(rs)
## Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=rs)

feature_epoch = {}

loo = LeaveOneOut()
X_loo = X_train
Y_loo = y_train

for fold, (train_idx, test_idx) in enumerate(loo.split(X_loo, Y_loo)):
    print('Current fold in loo: ', fold)
    print('\n')

    X_loo_train, X_loo_test = X_loo.iloc[train_idx], X_loo.iloc[test_idx]
    Y_loo_train, Y_loo_test = Y_loo.iloc[train_idx], Y_loo.iloc[test_idx]
    test_sample = Y_loo_test.index.tolist()[0]
    print('Samples for test: ', test_sample)  # 

    rf_model = RandomForestClassifier(random_state=82)
    rf_model.fit(X_loo_train, Y_loo_train)

    feature_importances = pd.Series(rf_model.feature_importances_, index=list(X_train.columns))
    threshold = np.percentile(feature_importances, 100 - top_percent)

    top_features = feature_importances[(feature_importances >= threshold) & (feature_importances > 0)].index.tolist()
    print('Feature count: ', len(top_features))
    print('Top features: ', top_features)
    feature_epoch[fold] = top_features


count_dict = {}
for value in feature_epoch.values():
    for sub_values in value:
        if sub_values in count_dict:
            count_dict[sub_values] += 1
        else:
            count_dict[sub_values] = 1
feature = []

for value, count in count_dict.items():
    if count > math.ceil(X_loo.shape[0]*freq_thr):   
        feature.append(value)
        print(f'{value} was selected in {count} iterations.')

print('Final selected feature count: ', len(feature))
print('Final selected features: ', feature)

## Selected features in training and test set
train_ = X_train[feature]
test_ = X_test[feature]

clf.fit(train_, y_train)
best_clf = clf.best_estimator_

## Prediction in train set
train_pred_label = best_clf.predict(train_)
train_accuracy = accuracy_score(y_train, train_pred_label)
predicted_probabilities_train = best_clf.predict_proba(train_)[:, 1]

## ROC in train set
predicted_probabilities_array_train = pd.DataFrame(predicted_probabilities_train)
predicted_probabilities_array_train.index = train_.index
fpr_train, tpr_train, thresholds = roc_curve(y_train, predicted_probabilities_array_train)
roc_auc_train = auc(fpr_train, tpr_train)

## Prediction in test set
test_pred_label = best_clf.predict(test_)
test_accuracy = accuracy_score(y_test, test_pred_label)
predicted_probabilities_test = best_clf.predict_proba(test_)[:, 1]  # 

## ROC in test set
predicted_probabilities_array_test = pd.DataFrame(predicted_probabilities_test)
predicted_probabilities_array_test.index = test_.index
fpr_test, tpr_test, thresholds = roc_curve(y_test, predicted_probabilities_array_test)
roc_auc_test = auc(fpr_test, tpr_test)

## Plot
plt.figure()
plt.plot(fpr_train, tpr_train, color='darkorange', lw=lw, label='ROC curve in training set (area = %0.3f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='blue', lw=lw, label='ROC curve in test set (area = %0.3f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('./ROCinTrain-Test.pdf', format="pdf",bbox_inches='tight')
plt.show()
    
## Save sample names in train set
file_path_train_sample = 'selected_train_samples.tsv'
print('Samples in training set are saved in: '+file_path_train_sample+'!!')
with open(file_path_train_sample, "w") as file:
    file.writelines(item + "\n" for item in X_train.index)
## Save sample names in test set
file_path_test_sample = 'selected_test_samples.tsv'
print('Samples in test set are saved in: '+file_path_test_sample+'!!')
with open(file_path_test_sample, "w") as file:
    file.writelines(item + "\n" for item in X_test.index)
## Save probabilities in train set
file_path_train_prob = 'train_probabilities.tsv'
print('The predicted probabilities of training set are saved in: '+file_path_train_prob+'!!')
predicted_probabilities_array_train.to_csv(file_path_train_prob, index=True,sep='\t')
## Save probabilities in test set
file_path_test_prob = 'test_probabilities.tsv'
print('The predicted probabilities of test set are saved in: '+file_path_test_prob+'!!')
predicted_probabilities_array_test.to_csv(file_path_test_prob, index=True,sep='\t')
## Save selected features
file_path_feature = 'selected_features.tsv'
print('Selected features are saved in: '+file_path_feature+'!!')
with open(file_path_feature, "w") as file:
    file.writelines(item + "\n" for item in feature)
## Save AUC in train and test
file_path_auc = 'AUCs.tsv'
print('AUCs of training and test set are saved in: '+file_path_auc+'!!')
with open(file_path_auc, "w") as file:
    file.writelines(f'Train Accuracy = {train_accuracy:.4f}\n')
    file.writelines(f'Train AUC = {roc_auc_train:.4f}\n')
    file.writelines(f'Test Accuracy = {test_accuracy:.4f}\n')
    file.writelines(f'Test AUC = {roc_auc_test:.4f}\n')
