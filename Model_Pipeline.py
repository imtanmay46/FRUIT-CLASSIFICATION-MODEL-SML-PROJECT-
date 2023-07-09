#Importing the Dependencies
import numpy as num
import pandas as pan
import matplotlib.pyplot as mtp
import seaborn as sea
from sklearn import *
from sklearn.cluster._kmeans import KMeans
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.decomposition import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

#Loading the Train.csv file for Model Training
gen_data = pan.read_csv('/Users/varul18/Desktop/2021569_TanmaySingh_SML_Project/train.csv')
gen_data = gen_data.drop(columns='ID', axis=1)

#Label Encoding the 'string' type target column
label=LabelEncoder()
gen_data['category'] = label.fit_transform(gen_data['category'])

#Outlier Detection & Removal
lof = LocalOutlierFactor(n_neighbors=10)
outlier_scores = lof.fit_predict(gen_data)

outliers = num.where(outlier_scores == -1)[0]
gen_data = gen_data.drop(index=outliers)

#Splitting the dataframe into 2 -> Data vS Target Variable
gen_data_x = gen_data.drop(['category'], axis=1, inplace=False)
gen_data_y = gen_data['category']

#Splitting the Train Set into Training & Validation Sets with 80:20 ratio
x_train, x_val, y_train, y_val = train_test_split(gen_data_x, gen_data_y, test_size=0.2, random_state=42)

#Dimensionality Reduction Algorithms such as PCA & LDA are used prior to model training
pca = PCA(n_components=470)
x_pca = pca.fit_transform(x_train)

lda = LinearDiscriminantAnalysis(n_components=19)
x_lda = lda.fit_transform(x_pca, y_train)

#Clustering was done to generate additional labels for the dataset before model training
kmeans = KMeans(n_clusters=20)
kmeans.fit(x_lda)
clustered_x_train = num.column_stack((x_lda, kmeans.labels_))

#Creating a Model Pipeline and testing the performance of each model on the given dataset
model_pipeline = []

model_pipeline.append(LogisticRegression(solver='newton-cg', max_iter=5000, random_state=42, n_jobs=-1))
model_pipeline.append(DecisionTreeClassifier(max_depth=10))
model_pipeline.append(RandomForestClassifier(max_depth=100))
model_pipeline.append(GaussianNB(var_smoothing=1e-11))
model_pipeline.append(KNeighborsClassifier(n_neighbors=101))
model_pipeline.append(LogisticRegressionCV(solver='newton-cg',max_iter=10000))
model_pipeline.append(AdaBoostClassifier(base_estimator=LogisticRegression(solver='newton-cg', max_iter=5000, random_state=42, n_jobs=-1), n_estimators=500, learning_rate=1.5, random_state=42))
model_pipeline.append(MLPClassifier(hidden_layer_sizes=(310,), activation='logistic', solver='lbfgs', max_iter=10000, learning_rate_init=1.75, random_state=42))

model_list=['Logistic Regression', 'Decision Trees', 'Random Forest', 'Naive Bayes', 'KNN', 'Logistic Regression CV', 'Adaboost', 'MLP']

#Arrays to store the results of Accuracy, AUC Value, Confusion Matrix associated with each model in the Model Pipeline respectively
acc_list=[]
auc_list=[]
cm_list=[]

for model in model_pipeline:
    #Training the Model
    model.fit(clustered_x_train, y_train)

    #Preparing the Data for validation
    x_pca_val = pca.transform(x_val)
    x_lda_val = lda.transform(x_pca_val)
    x_lda_val_clustered = num.column_stack((x_lda_val, kmeans.predict(x_lda_val)))

    #Predictions made by the model
    y_pred=model.predict(x_lda_val_clustered)

    #Storing the performance statistics of each model in the Model Pipeline into the respective arrays
    acc_list.append(metrics.accuracy_score(y_val, y_pred))
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred, pos_label=model.classes_[1])
    auc_list.append(round(metrics.auc(fpr, tpr), 2))
    cm_list.append(confusion_matrix(y_val, y_pred))

# Visualising the Confusion Matrix for each model in the Model Pipeline
fig, axes = mtp.subplots(nrows=2, ncols=4, figsize=(18,12))

for i, ax in enumerate(axes.flat):
    cm = cm_list[i]
    model = model_list[i]
    cm_plot = sea.heatmap(cm, annot=True, cmap='Blues_r', ax=ax)
    cm_plot.set_title(model)
    cm_plot.set_xlabel('Predicted Values')
    cm_plot.set_ylabel('Actual Values')

mtp.show()

#Creating a Dataframe for Results & displaying it on the console to check which model performs best for the given dataset
result_df = pan.DataFrame({'Model' : model_list, 'Accuracy': acc_list, 'AUC': auc_list})
print(result_df)