import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc

def value__counts(df_):
    """it takes a df and return value_counts for all the features"""
    for i in df_.columns:
        print(df_[i].value_counts())
        
        
def count_freq_plot(df, col,name, want_percentages = False):
    """it retuns a bar plot with the percentages or w/o percentages"""
    
    sns.set(style="darkgrid")
    plt.figure(figsize=(8,6))
    total = float(len(df)) 
    ax = sns.countplot(x=col, data=df, dodge =False) # for Seaborn version 0.7 and more
    ax.set(xlabel=col, title= f'Count Frequency: Positive {name} vs Negative {name} ')


    if want_percentages == True:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2., height + 3,'{:1.2%}'.format(height/total),ha="center") 

        plt.show()

def normalizad_features(x):
    """Nornalized features """
    for col in x.columns:
        x[col] = (x[col]-min(x[col]))/ (max(x[col]) - min(x[col]))
    return x.head()


def split (independent_va, dependent_va):
    """return train, val, test set will be 60%, 20%, 20% of the dataset respectively"""
    X_train, X_test, y_train, y_test = train_test_split( independent_va , dependent_va, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    return X_train, y_train, X_test, y_test, X_val, y_val


def classifier_modeling (clf, x_tr, y_tr, x_vali):
    """return model and prediction value"""
    clf1 = clf # create the model
    model_1 = clf1.fit(x_tr, y_tr)
    y_pred = clf1.predict(x_vali)# make a prediction
    return model_1, y_pred
    

def metrix_classifier(x , y):
    """calculated the accuracy and print out accuracy and classification report"""
    accuracy = round(metrics.accuracy_score(x, y) * 100 ,2)
    print(f"Accuracy: {accuracy} %")
    precision,recall,fscore,support = metrics.precision_recall_fscore_support(x,y, average = 'weighted')
    print('Precision_weighted: ', str(round(precision*100,2)),"%")
    print('Recall_weighted: ', round(recall *100, 2),"%")
    print('fscore_weightted: ', round(fscore* 100, 2),"%")
    print(classification_report(x,  y))
    

def confusion_m(x,y,classifier):
    """it plots a confusion matrix """
    cm = confusion_matrix(x, y)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix ' + classifier)
    ax.xaxis.set_ticklabels(['No_buy', 'Buy']); ax.yaxis.set_ticklabels(['No_buy', 'Buy'])

    
def tune_parameters_clf( model, n, X_train, y_train, X_val, param_grid_1={}):
    """return the best parameteres for the model and the prediction value"""

    grid_clf = GridSearchCV(model, param_grid_1, scoring='accuracy', cv=n, n_jobs=1)
    grid_clf.fit(X_train, y_train)

    best_parameters = grid_clf.best_params_

    print("Grid Search found the following optimal parameters: ")
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    val_preds = grid_clf.predict(X_val)
    return val_preds

def auc_plot(y, val):
    """it takes output value for y and the predicted value returns a ROC plot"""
    fpr, tpr, thresholds = roc_curve(y, val)
    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def ploting_coefficients(model, columns_n, model_name):
    """it takes model coefficients, a list of the name of col and model name and return a plot in ASC order"""

    cdf = pd.DataFrame(model, columns_n, columns=['Coefficients']) # store coefficients in a df
    cdf['Features'] = cdf.index # conver the index into a col
    cdf.sort_values(['Coefficients'], inplace = True , ascending = False) # sorting the coefficients
    plt.figure(figsize=(20,10))
    ax = sns.barplot(x="Coefficients", y="Features", data=cdf)
    ax.set(title = f'Important Attributes in {model_name}')
    plt.show()
