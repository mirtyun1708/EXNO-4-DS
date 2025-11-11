# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method



# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("income(1).csv",na_values=[" ?"])
data
```

**Output:**

<img width="1620" height="694" alt="image" src="https://github.com/user-attachments/assets/0ee16e02-5fb9-4f14-905f-634ec3d6f6ec" />

```
data.isnull().sum()
```

**Output:**

<img width="353" height="711" alt="image" src="https://github.com/user-attachments/assets/782c75fa-256a-4d79-b6d2-317b6a2b8125" />

```
missing = data[data.isnull().any(axis=1)]
missing
```

**Output:**

<img width="1769" height="609" alt="image" src="https://github.com/user-attachments/assets/26b13796-530c-4f97-be2e-69a3b2a81a87" />

```
data2 = data.dropna(axis=0)
data2
```

**Output:**

<img width="1759" height="784" alt="image" src="https://github.com/user-attachments/assets/ab591719-1e76-4815-a129-787efceadfdd" />

```
sal = data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

**Output:**

<img width="1076" height="372" alt="image" src="https://github.com/user-attachments/assets/c46f3f3e-c4aa-40d3-b170-52e16ce2db8b" />

```
sal2 = data2['SalStat']
dfs = pd.concat([sal,sal2],axis=1)
dfs
```

**Output:**

<img width="558" height="650" alt="image" src="https://github.com/user-attachments/assets/f0748006-5b8e-4075-9051-1af470bda221" />

```
data2
```

**Output:**

<img width="1653" height="579" alt="image" src="https://github.com/user-attachments/assets/ad9234c0-95c4-47aa-81fb-a43cdfe54aa5" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

**Output:**

<img width="1769" height="692" alt="image" src="https://github.com/user-attachments/assets/b6d12701-1cd5-49f0-8925-f6ac8425fbed" />

```
columns_list = list(new_data.columns)
print(columns_list)
```

**Output:**

<img width="1133" height="136" alt="image" src="https://github.com/user-attachments/assets/82689640-4333-4f9b-ac27-f69821d5fe1e" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

**Output:**

<img width="1081" height="142" alt="image" src="https://github.com/user-attachments/assets/e9e641c7-978b-4825-a429-41b1b9c582ef" />

```
y = new_data['SalStat'].values
print(y)
```

**Output:**

<img width="422" height="119" alt="image" src="https://github.com/user-attachments/assets/8f1a8ca8-1701-48d8-bbfe-b7efab445e89" />

```
x=new_data[features].values
x
```

**Output:**

<img width="759" height="247" alt="image" src="https://github.com/user-attachments/assets/c27834d5-c7e3-4e9a-8a74-aa0bfc7c126d" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```

**Output:**

<img width="890" height="219" alt="image" src="https://github.com/user-attachments/assets/7ab5e446-6423-4b4e-ac37-20a304e3b501" />

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix = confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

**Output:**

<img width="601" height="191" alt="image" src="https://github.com/user-attachments/assets/5ff31fea-4225-4f06-8b37-851862960c63" />

```
accuracy = accuracy_score(test_y,prediction)
print(accuracy)
```

**Output:**

<img width="511" height="140" alt="image" src="https://github.com/user-attachments/assets/fe9d90f9-5e86-47ba-96d8-630f0d31935a" />

```
print("Misclasssifed Samples :%d"% (test_y != prediction).sum())
```

**Output:**

<img width="697" height="118" alt="image" src="https://github.com/user-attachments/assets/09cc616d-1cdd-46f3-9ce2-2918435ba18a" />

```
data.shape
```

**Output:**

<img width="281" height="89" alt="image" src="https://github.com/user-attachments/assets/2e5aee82-d44c-49ef-9218-eba6a102c53b" />

```
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

data = {
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
selector.fit(x, y) # Fit the selector to the data
selected_feature_indices=selector.get_support(indices=True)

selectedfeature=x.columns[selected_feature_indices]
print("Selected Features:")
print(selectedfeature)

```

**Output:**

<img width="1054" height="568" alt="image" src="https://github.com/user-attachments/assets/82f1b5ab-d097-4e42-839f-69f9ea8f393a" />

```
from scipy.stats import chi2_contingency

tips = sns.load_dataset('tips')
tips.head()
```

**Output:**

<img width="684" height="404" alt="image" src="https://github.com/user-attachments/assets/9e11f1ff-82c3-4658-9a98-73161ff0a697" />

```
tips.time.unique()
```

**Output:**

<img width="571" height="112" alt="image" src="https://github.com/user-attachments/assets/2d83c872-4414-4925-beb2-6d4242baf690" />

```
contingency_table = pd.crosstab(tips['sex'],tips['time'])
contingency_table
```

**Output:**

<img width="710" height="252" alt="image" src="https://github.com/user-attachments/assets/cd0990cb-682e-44d7-833b-ea7b79805ed6" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
print('Chi-square statistic:',chi2)
print('p-value:',p)
```

**Output:**

<img width="549" height="162" alt="image" src="https://github.com/user-attachments/assets/40adab07-c99d-498d-abc2-6ada90db71c1" />

# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed.

# SUMMARY:
SUMMARY

In this experiment, the dataset was read, cleaned, and missing values were handled. Categorical features were converted to numerical form using one-hot encoding. Feature scaling techniques were applied to bring all features to a similar scale. Important features were selected using feature selection methods to improve model performance. Finally, a K-Nearest Neighbors (KNN) classifier was used, and the model showed good accuracy in predicting the target variable.
