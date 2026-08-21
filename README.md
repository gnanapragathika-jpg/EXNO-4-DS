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
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
<img width="285" height="202" alt="image" src="https://github.com/user-attachments/assets/a446ed9f-2074-4552-affe-8a8e389fa28d" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="133" height="117" alt="image" src="https://github.com/user-attachments/assets/771199d1-1abd-49a2-8abb-7af0a8788513" />

```
df.dropna()
```
<img width="301" height="433" alt="image" src="https://github.com/user-attachments/assets/948aa9dd-84c5-4eb6-9fb7-11790adddea3" />

   ```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
<img width="145" height="77" alt="image" src="https://github.com/user-attachments/assets/400b90de-c8f9-4bd4-beac-dde833299300" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```
<img width="275" height="195" alt="image" src="https://github.com/user-attachments/assets/fb287f9c-e17e-439e-86ee-8aae66d381af" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="322" height="356" alt="image" src="https://github.com/user-attachments/assets/a3d12ef4-f37e-4931-aa10-b82af636c413" />

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="311" height="354" alt="image" src="https://github.com/user-attachments/assets/445877c8-53bd-4c1a-bae3-d1688e7219eb" />

```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
```
<img width="286" height="200" alt="image" src="https://github.com/user-attachments/assets/cca78441-1f7b-4cd8-9ef4-9e8dcf0de87f" />

```
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
<img width="330" height="428" alt="image" src="https://github.com/user-attachments/assets/1868415d-8ba3-406b-b027-cf81d29a12f4" />

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
<img width="317" height="195" alt="image" src="https://github.com/user-attachments/assets/ddc93895-8cf5-419b-bf7f-910492781fb7" />

```
df=pd.read_csv("income(1) (1).csv")
df.info()
```
<img width="415" height="431" alt="image" src="https://github.com/user-attachments/assets/1a869178-c8eb-4438-b434-8e18d6fde543" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="189" height="303" alt="image" src="https://github.com/user-attachments/assets/fee41cbc-6695-4764-b982-8f0e86afe116" />

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="871" height="430" alt="image" src="https://github.com/user-attachments/assets/4059b730-c9f4-4753-8556-a9dc3ede76de" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="736" height="423" alt="image" src="https://github.com/user-attachments/assets/63f97fa2-82df-4689-bdc8-e749e3dd5ef4" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="345" height="69" alt="image" src="https://github.com/user-attachments/assets/ad22fc6a-5a76-44c7-b0f3-abdf1ae224be" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv")
df.info()
```
<img width="410" height="425" alt="image" src="https://github.com/user-attachments/assets/63acfabc-3a2f-4cb0-94dd-d4dbecac97ef" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="866" height="427" alt="image" src="https://github.com/user-attachments/assets/ba08bc0f-131f-409d-826e-3ce25df9476b" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="742" height="418" alt="image" src="https://github.com/user-attachments/assets/d4c2cf47-a5a4-4038-adb5-a89b5c88c3f5" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="736" height="90" alt="image" src="https://github.com/user-attachments/assets/ecd6c769-711b-4243-8260-1b9d959f1449" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="392" height="30" alt="image" src="https://github.com/user-attachments/assets/9c088d77-eaf5-41d7-8b41-4281b01a84ed" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="561" height="27" alt="image" src="https://github.com/user-attachments/assets/f727034c-52f0-439f-9106-6d0cc019fcbc" />

```
!pip install skfeature-chappers
```
<img width="1234" height="476" alt="image" src="https://github.com/user-attachments/assets/a157524c-65e2-4076-b8af-537d9eae7877" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = ['JobType','EdType','maritalstatus','occupation','relationship','race','gender','nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="741" height="427" alt="image" src="https://github.com/user-attachments/assets/4d4cde27-907c-4e58-9cd4-9487cb9ea8dc" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="846" height="56" alt="image" src="https://github.com/user-attachments/assets/eb77ac28-1363-4a07-82c0-2b0cb5a4a475" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()

n_features_to_select =6

rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
<img width="1252" height="669" alt="image" src="https://github.com/user-attachments/assets/9a10e118-d283-4750-81e4-8579775173af" />

# RESULT:
Thus feature scaling and feature selection is performed successfully.
