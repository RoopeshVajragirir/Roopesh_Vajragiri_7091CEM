import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, neighbors, tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import *
import graphviz
import IPython as ip
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

wine=load_wine()
print("Target",wine.target_names)
print("Features",wine.feature_names)

#Data Collection
#Read red wine data from red_wine_quality.csv file
red_wine_CSVdata = pd.read_csv("Red_Wine_Quality.csv")

red_wine_CSVdata.shape

#Print 1st 20 rows of data from red_wine_quality.csv file
print(red_wine_CSVdata.head(20))

# print the information about red_wine_quality.csv file
red_wine_CSVdata.info()

#Data Inspection
print(red_wine_CSVdata.isnull().sum())

#Data analysing and visualisation
red_wine_CSVdata.describe()
red_wine_CSVdata['Quality'].unique()
red_wine_CSVdata['Quality'].value_counts()
red_wine_CSVdata['Quality'].count()


#Count plot to display no. of values for each quality
sb.countplot(x='Quality',data = red_wine_CSVdata)
red_wine_CSVdata1 = sb.countplot(x='Quality',data = red_wine_CSVdata)
for p in red_wine_CSVdata1.patches:
    red_wine_CSVdata1.annotate('{:1.1f}'.format(p.get_height()),(p.get_x()+0.25,p.get_height()+0.01))
#plt.show()



#Display the barplot of Quality vs (Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugars,-
# Chlorides, Free Sulphur Dioxide, Total Sulphur Dioxide, Density, pH, Sulphates, Alcohol)
headers = pd.read_csv("Red_Wine_Quality.csv",index_col=0, nrows=0)
print("Count of the Headers are ",len(headers.columns))
n = len(headers.columns)
for i in range(n-1):
    plot = plt.figure(figsize=(5,5),num=  "Quality Vs "+ headers.columns[i])
    sb.barplot(x='Quality', y=headers.columns[i], data = red_wine_CSVdata)
    #plt.show()




#Display the boxplot of Quality vs (Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugars,-
# Chlorides, Free Sulphur Dioxide, Total Sulphur Dioxide, Density, pH, Sulphates, Alcohol)
for i in range(n-1):
    plt.figure(figsize=(10,5),num=  "Quality Vs "+ headers.columns[i])
    sb.boxplot(x = 'Quality', y = headers.columns[i], data = red_wine_CSVdata)
    axis = plt.gca()
    axis.set_title("Quality Vs "+ headers.columns[i])
    #plt.show()




#Display the lmplot of Quality vs (Fixed Acidity, Volatile Acidity, Citric Acid, Residual Sugars,-
# Chlorides, Free Sulphur Dioxide, Total Sulphur Dioxide, Density, pH, Sulphates, Alcohol)
for i in range(n-1):
    sb.lmplot(x='Quality', y=headers.columns[i], data = red_wine_CSVdata)
    axis = plt.gca()
    axis.set_title("Quality Vs "+ headers.columns[i])
    #plt.show()


#Display the heatmap
plt.figure(figsize=(15,15),num=" Heat Map for Red Wine Quality")
sb.heatmap(red_wine_CSVdata.corr(),cmap="Blues", annot=True)
#plt.show()




#Data Preprocessing and removing the unwanted attributes
x_Value = red_wine_CSVdata.drop('Quality',axis=1)
y_Value = red_wine_CSVdata['Quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)


#Train and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(x_Value,y_Value,test_size=0.2, random_state=2)


#Model Training
#Supervised Machine Learning Techinque: Decision Tree - Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)


#Model Evaluation
X_test_prediction = model.predict(X_test)


#Accuracy Score
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)



#Scaling the Red Wine Data to take account of variations in Mean and Standard Deviations
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

def models(X_train,Y_train):
    logReg = linear_model.LogisticRegression(random_state = 0)
    logReg.fit(X_train, Y_train)

    knn = neighbors.KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
    knn.fit(X_train, Y_train)

    svc_lin = SVC(kernel='linear',random_state=0)
    svc_lin.fit(X_train,Y_train)

    svc_rbf = SVC(kernel='rbf',random_state=0)
    svc_rbf.fit(X_train,Y_train)

    gauss = GaussianNB()
    gauss.fit(X_train,Y_train)

    decisionTree = DecisionTreeClassifier(criterion='entropy',random_state=0)
    decisionTree.fit(X_train,Y_train)

    forest = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
    forest.fit(X_train,Y_train)

    print('Logistic Regression Training Accuracy [0]',logReg.score(X_train,Y_train))
    print('K Neighbors Classifier Training Accuracy [1]',knn.score(X_train,Y_train))
    print('Support Vector Machine (Linear Classifier) Training Accuracy [2]',svc_lin.score(X_train,Y_train))
    print('Support Vector Machine (RBF Classifier) Training Accuracy [3]',svc_rbf.score(X_train,Y_train))
    print('Gaussian Naive Bayes Training Accuracy [4]',gauss.score(X_train,Y_train))
    print('Decision Tree Classifier Training Accuracy [5]',decisionTree.score(X_train,Y_train))
    print('Random Forest Classifier Training Accuracy [6]',forest.score(X_train,Y_train))
    return logReg,knn,svc_lin,svc_rbf,gauss,decisionTree,forest





#Evaluating Performance on Training Sets
model = models(X_train, Y_train)


#Evaluating Performance on Training Sets with Confusion Matrix
for i in range(len(model)):
    print("______________________________________________________________")
    print(model[i])
    print("______________________________________________________________")
    print(classification_report(Y_test,model[i].predict(X_test)))
    print("Mean Squared Error is : ",mean_squared_error(Y_test,model[i].predict(X_test)))
    print('R-squared score: ',r2_score(Y_test,model[i].predict(X_test)))
    plt.figure(figsize=(12,8))
    pred = model[i].predict(X_test)
    cm = confusion_matrix(Y_test,pred)
    ax=plt.subplot()
    sb.heatmap(cm,annot=True,fmt='g',ax=ax,cmap="Blues")
    plt.show()
    
num_folds = 10
results = []
names = []
models_list = []
models_list.append(('Decision Tree', DecisionTreeClassifier()))
models_list.append(('Logistic', linear_model.LogisticRegression())) 
models_list.append(('SVC', SVC()))
models_list.append(('Random Forest', RandomForestClassifier()))
models_list.append(('GaussianNB', GaussianNB()))
models_list.append(('KNeighbors', neighbors.KNeighborsClassifier()))

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=123,shuffle=True )
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

#Box plot for comparing the performance among the models
fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#Creating Classification Bins
classBins = (2,6.5,8)
quality = ['bad', 'good']
red_wine_CSVdata['Quality'] = pd.cut(red_wine_CSVdata['Quality'], bins = classBins, labels = quality)
red_wine_CSVdata.describe()
label_quality = preprocessing.LabelEncoder()
red_wine_CSVdata['Quality'] = label_quality.fit_transform(red_wine_CSVdata['Quality'])



#Decision Tree for Red Wine Quality
clf = tree.DecisionTreeClassifier(max_depth = 2, random_state =0)
clf.fit(X_train,Y_train)
clf.predict(X_test)
tree.plot_tree(clf)
fn=['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugars','Chlorides', 'Free Sulphur Dioxide', 'Total Sulphur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol','Quality']
fig,axes = plt.subplots(nrows = 1,ncols =1,figsize=(4,4),dpi=300)
tree.plot_tree(clf,class_names = wine.target_names, feature_names =fn,filled= True)
fig.savefig('Red Wine Quality Decision Tree.png')


#Predicting the Wine Quality using Naive Bayes
model1 = GaussianNB()
model1.fit(X_train, Y_train)
X_test_prediction = model1.predict(X_test)

input_data =[7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4]
input_data_array = np.asarray(input_data)
input_reshaped_data = input_data_array.reshape(1,-1)
prediction = model1.predict(input_reshaped_data)
if(prediction[0] == 0):
    print("______________________________________________________________")
    print('Good Quality Wine')
    print("______________________________________________________________")
else:
    print("______________________________________________________________")
    print('Bad Quality Wine')
    print("______________________________________________________________")
