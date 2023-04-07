import numpy as num
import pandas as pan
import mplcyberpunk as mcy
import matplotlib.pyplot as mtp
import seaborn as sea
import math
import random
from sklearn import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.decomposition import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

gen_data=pan.read_csv('/Users/varul18/Desktop/SML-Project/train.csv')
gen_data=gen_data.dropna()
gen_data=gen_data.drop_duplicates()


unavailable_data=pan.DataFrame({'Total Unavailable Data': gen_data.isnull().sum(), 'Missing Info': (gen_data.isnull().sum()/random.randint(1000,90000))*100})

label=LabelEncoder()
gen_data['category']=label.fit_transform(gen_data['category'])
# print(gen_data)

x_coordinate = gen_data.drop(['category'], axis=1, inplace=False)
y_coordinate = gen_data['category']

x_coordinate=x_coordinate.drop_duplicates()
x_coordinate=x_coordinate.dropna()

scaler=StandardScaler()
X_Scaled=scaler.fit_transform(x_coordinate)

pca=PCA(n_components=78)
X_PCA=pca.fit_transform(X_Scaled)

lda=LinearDiscriminantAnalysis()
X_LDA=lda.fit_transform(X_PCA, y_coordinate)

iso_forest = IsolationForest(n_estimators=1000, contamination=1e-6, random_state=42)
outliers = iso_forest.fit_predict(X_LDA)
X_LDA = X_LDA[outliers != -1]
y_coordinate = y_coordinate[outliers != -1]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_LDA, y_coordinate, test_size=0.2, random_state=42)

base_estimator = DecisionTreeClassifier(max_depth=10)
Modelling = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=1850, learning_rate=1.5, random_state=True)
Modelling.fit(X_train, y_train)

# Predict on validation set
y_pred = Modelling.predict(X_val)

print("\nPredicted Data Values of 'y' \n")
print(y_pred)
print("\n\n")

accuracy = accuracy_score(y_pred, y_val)
print("Accuracy on validation set:", accuracy*100)

Testing_data=pan.read_csv('/Users/varul18/Desktop/SML-Project/test.csv')
Testing_data=Testing_data.dropna()
Testing_data=Testing_data.drop_duplicates()

# Transform testing data using same steps as training data
Testing_data_scaled = scaler.transform(Testing_data.drop_duplicates().dropna())
Testing_data_pca = pca.transform(Testing_data_scaled)
Testing_data_lda = lda.transform(Testing_data_pca)

# Predict on testing data
y_pred_test = Modelling.predict(Testing_data_lda)

# Create dataframe with ID and predicted category values
result_df = pan.DataFrame({'ID': range(len(y_pred_test)), 'category': label.inverse_transform(y_pred_test)})

# Save dataframe to csv
result_df.to_csv('/Users/varul18/Desktop/SML-Project/new_predictions.csv', index=False)

print("\nPredicted Data Values of 'y' for testing data\n")
print(y_pred_test)
print(label.inverse_transform(y_pred_test))
print("\n\n")