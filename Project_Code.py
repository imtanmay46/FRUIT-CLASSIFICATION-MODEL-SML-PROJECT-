#Importing the Dependencies
import numpy as num
import pandas as pan
from sklearn import *
from sklearn.cluster._kmeans import KMeans
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.decomposition import *
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

#Using the 'LogisticRegressionCV' model with 'newton-cg' solver for model training (This was done after verifying it's performance in the model pipeline(Present in a separate code/file))
Modelling = LogisticRegressionCV(solver='newton-cg',max_iter=10000)
Modelling.fit(clustered_x_train, y_train)

#Checking the accuracy in the Training Set
print("\nAccuracy on Training Set: ", accuracy_score(y_train, Modelling.predict(clustered_x_train)))

#Using K-Fold Cross Validation to verify model performance on the Validation Set
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

#Preparing the Validation Set for K-Fold Cross Validation
x_pca_val = pca.transform(x_val)
x_lda_val = lda.transform(x_pca_val)
x_val_clustered = num.column_stack((x_lda_val, kmeans.predict(x_lda_val)))

#Storing the scores from the Cross Validation
scores = cross_val_score(Modelling, x_val_clustered, y_val, cv=k_fold, scoring='accuracy')

#Checking the Accuracy of the trained ML model on the Validation Set
print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = Modelling.predict(x_val_clustered)

accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy on validation set:", accuracy*100)

#Preparing the entire training set from Train.csv file for model re-training
gen_data_x_pca = pca.transform(gen_data_x)
gen_data_x_lda = lda.transform(gen_data_x_pca)
gen_data_x_clustered = num.column_stack((gen_data_x_lda, kmeans.predict(gen_data_x_lda)))
Modelling.fit(gen_data_x_clustered, gen_data_y)

#Loading the testing dataset and performing the same transformations as the training dataset
testing_data = pan.read_csv('/Users/varul18/Desktop/2021569_TanmaySingh_SML_Project/test.csv')
testing_data = testing_data.drop(columns='ID', axis=1)

testing_data_pca = pca.transform(testing_data)
testing_data_lda = lda.transform(testing_data_pca)
testing_data_clustered = num.column_stack((testing_data_lda, kmeans.predict(testing_data_lda)))

#Predictions made on the testing dataset
y_pred_test = Modelling.predict(testing_data_clustered)

#Generating a new csv file with 'ID', 'Predictions' being the two columns
result_df = pan.DataFrame({'ID': range(len(y_pred_test)), 'category': label.inverse_transform(y_pred_test)})
result_df.to_csv('/Users/varul18/Desktop/2021569_TanmaySingh_SML_Project/predictions.csv', index=False)