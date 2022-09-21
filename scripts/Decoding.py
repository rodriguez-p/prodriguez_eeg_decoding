# Individual, group and temporal decoding tutorials (https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/ApplyingMachineLearningMethods_1.html)

# %%# Load libraries
import mne
from mne.decoding import Vectorizer

from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from os.path import isfile, join
from os import listdir

import statistics
# %% 
DATA_ROOT = 'C:/Users/Profa. Ana Chica/Desktop/prodriguez_eeg_decoding/data'
FILE_PREFIX = 'PRODRIGUEZ_'
MONTAGE_FILE = 'easycap-M10.txt'
    
#%% Load epoched data and select epochs of interest
def individual_decoding(id):
    SUBJECT = id
    epochs = mne.read_epochs("{}/S{}/{}{:06d}-epo.fif".format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT))
    
    #combine Present and Absent events
    epochs_present = mne.epochs.combine_event_ids(epochs['Present/Seen', 'Present/Unseen'], ['Present/Seen', 'Present/Unseen'], {'Present': 100})
    epochs_absent = mne.epochs.combine_event_ids(epochs['Absent/Seen', 'Absent/Unseen'], ['Absent/Seen', 'Absent/Unseen'], {'Absent': 101})
    
    epochs = mne.concatenate_epochs([epochs_present, epochs_absent], add_offset=True)
    epochs_PA = epochs['Present', 'Absent']
    print(epochs_PA)
    
    data_PA = epochs_PA.get_data()
    labels_PA = epochs_PA.events[:,-1]
    
    # Split dataset into training/test (70/30)
    train_data_PA, test_data_PA, labels_train_PA, labels_test_PA = train_test_split(data_PA, labels_PA, test_size=0.3, random_state=42)
    
    # SVM
    clf_svm_pip = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(random_state=42))
    parameters = {'svc__kernel':['linear', 'rbf', 'sigmoid'], 'svc__C':[0.1, 1, 10]}
    
    #GridSearchCV will try to find the best performing parameters
    gs_cv_svm = GridSearchCV(clf_svm_pip, parameters, scoring='accuracy', cv=StratifiedKFold(n_splits=5), return_train_score=True)
    gs_cv_svm.fit(train_data_PA, labels_train_PA)
    print('Best Parameters (SVM): {}'.format(gs_cv_svm.best_params_))
    print('Best Score (SVM): {}'.format(gs_cv_svm.best_score_))
    
    #Prediction
    predictions_svm = gs_cv_svm.predict(test_data_PA)
    
    #Evaluate
    report_svm = classification_report(labels_test_PA, predictions_svm, target_names=['Present', 'Absent'])
    print('SVM Clasification Report:\n {}'.format(report_svm))
    
    acc_svm = accuracy_score(labels_test_PA, predictions_svm)
    print("Accuracy of SVM model: {}".format(acc_svm))
    
    precision_svm,recall_svm,fscore_svm,support_svm=precision_recall_fscore_support(labels_test_PA,predictions_svm,average='macro')
    print('Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_svm,recall_svm,fscore_svm))
    
    lines_SVM = ['Best Parameters (SVM): {}'.format(gs_cv_svm.best_params_), 'Best Score (SVM): {}'.format(gs_cv_svm.best_score_), 'SVM Clasification Report:\n {}'.format(report_svm), "Accuracy of SVM model: {}".format(acc_svm), 'Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_svm,recall_svm,fscore_svm)]
    with open('{}/S{}/{}{:06d}_ClassificationParam.txt'.format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT), 'w') as f:
        for line in lines_SVM:
            f.write(line)
            f.write('\n')
        
    # Logistic Regression
    clf_lr_pip = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(random_state=42))
    parameters = {'logisticregression__penalty':['l1', 'l2']}
    gs_cv_lr = GridSearchCV(clf_lr_pip, parameters, scoring='accuracy')
    gs_cv_lr.fit(train_data_PA, labels_train_PA)
    
    print('Best Parameters (LR): {}'.format(gs_cv_lr.best_params_))
    print('Best Score (LR): {}'.format(gs_cv_lr.best_score_))
    
    #Predictions
    predictions_lr = gs_cv_lr.predict(test_data_PA)
    
    #Evaluation
    report_lr = classification_report(labels_test_PA, predictions_lr, target_names=['Present', 'Absent'])
    print('LR Clasification Report:\n {}'.format(report_lr))
    
    acc_lr = accuracy_score(labels_test_PA, predictions_lr)
    print("Accuracy of LR model: {}".format(acc_lr))
    
    precision_lr,recall_lr,fscore_lr,support_lr=precision_recall_fscore_support(labels_test_PA,predictions_lr,average='macro')
    print('Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_lr,recall_lr,fscore_lr))
    
    lines_LR = ['Best Parameters (LR): {}'.format(gs_cv_lr.best_params_), 'Best Score (LR): {}'.format(gs_cv_lr.best_score_), 'LR Clasification Report:\n {}'.format(report_lr), "Accuracy of LR model: {}".format(acc_lr), 'Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_lr,recall_lr,fscore_lr)]
    with open('{}/S{}/{}{:06d}_ClassificationParam.txt'.format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT), 'a') as f:
            f.write('\n' '\n'.join(lines_LR))
            
    # Linear Discriminant Analysis
    clf_lda_pip = make_pipeline(Vectorizer(), StandardScaler(), LinearDiscriminantAnalysis(solver='svd'))
    clf_lda_pip.fit(train_data_PA,labels_train_PA)
    
    #Predictions
    predictions_lda = clf_lda_pip.predict(test_data_PA)
    
    #Evaluation
    report_lda = classification_report(labels_test_PA, predictions_lda, target_names=['Present', 'Absent'])
    print('LDA Clasification Report:\n {}'.format(report_lda))
    
    acc_lda = accuracy_score(labels_test_PA, predictions_lda)
    print("Accuracy of LDA model: {}".format(acc_lda))
    
    precision_lda,recall_lda,fscore_lda,support_lda=precision_recall_fscore_support(labels_test_PA,predictions_lda,average='macro')
    print('Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_lda,recall_lda,fscore_lda))    
    
    lines_LDA = ['LDA Clasification Report:\n {}'.format(report_lda), "Accuracy of LDA model: {}".format(acc_lda), 'Precision: {0}, Recall: {1}, f1-score:{2}'.format(precision_lda,recall_lda,fscore_lda)]
    with open('{}/S{}/{}{:06d}_ClassificationParam.txt'.format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT), 'a') as f:
            f.write('\n' '\n'.join(lines_LDA))    
            
    # Plot accuracy values of the three models  #MODIFICAR PLOT PARA QUE SÓLO MUESTRE UNA TAREA
    accuracies, f1_scores = [], []
    accuracies.append([acc_svm, acc_lr, acc_lda])
    f1_scores.append([fscore_svm, fscore_lr, fscore_lda])
        
    #Plot Accuracies
    tasks = ['PA', 0, 0]
    labels = ['SVM', 'LR', 'LDA']
    plotEvalMetrics(tasks, labels, accuracies, 'Accuracy')
    print(accuracies)
    plt.savefig('{}/S{}/{}{:06d}_ClassificationACC.png'.format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT), bbox_inches='tight', edgecolor='w')
    plt.close()
    
    #Plot F1 Scores
    tasks = ['PA', 0, 0]
    labels = ['SVM', 'LR', 'LDA']
    plotEvalMetrics(tasks, labels, f1_scores, 'F1-Scores')
    print(f1_scores)
    plt.savefig('{}/S{}/{}{:06d}_ClassificationF1-Scores.png'.format(DATA_ROOT , SUBJECT, FILE_PREFIX, SUBJECT), bbox_inches='tight', edgecolor='w')
    plt.close()

def plotEvalMetrics(tasks, labels, evalMetric, metricName):
    width = 0.2  # the width of the bars

    # Set position of bar on X axis
    rects1 = np.arange(len(evalMetric[:][0]))
    rects2 = [x + width for x in rects1]
    rects3 = [x + width for x in rects2]

    plt.bar(rects1, list(zip(*evalMetric))[0], color='#87CEFA', width=width, edgecolor='white', label=labels[0])
    plt.bar(rects2, list(zip(*evalMetric))[1], color='#FFE4E1', width=width, edgecolor='white', label=labels[1])
    plt.bar(rects3, list(zip(*evalMetric))[2], color='#CD5C5C', width=width, edgecolor='white', label=labels[2])

    plt.xlabel('Classification Tasks')
    plt.xticks([r + width for r in range(len(evalMetric[:][0]))], tasks)
    plt.ylabel(metricName)

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', )
    plt.show()
    
#%%
for id in range(10, 24, 1):
    if id == 11:
        pass
    else:
        individual_decoding(id)
        
