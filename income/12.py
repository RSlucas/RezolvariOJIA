# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# %%
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# %%
df_train.info()

# %%
frame_tsk1 = df_train[(df_train['income'] == ">50K") & (df_train['native_country'] != 'United-States')]

# %%
frame_tsk1

# %%
sub1 = frame_tsk1['native_country'].mode()[0]

# %%
sub1

# %%
frame_tsk2 = df_train[(df_train['income'] == ">50K")]

# %%
total = df_train.groupby('occupation').size()

# %%
rata_mare = frame_tsk2.groupby('occupation').size()

# %%
rate = rata_mare/total

# %%
sub2 = rate.idxmax()

# %%
df_train.info()

# %%
df_train = df_train.drop(['relationship', 'race'], axis=1)
df_test = df_test.drop(['relationship', 'race'], axis=1)

# %%
df_test.select_dtypes(include='object').columns

# %%
catego =['workclass', 'education', 'marital_status', 'occupation', 'gender',
       'native_country']

# %%

df_train = pd.get_dummies(df_train,  columns=catego, drop_first=True)
df_test = pd.get_dummies(df_test, columns=catego, drop_first=True)

# %%
for col in df_train.select_dtypes(include='bool'):
    df_train[col] = df_train[col].astype(int)

for col in df_test.select_dtypes(include='bool'):
    df_test[col] = df_test[col].astype(int)



# %%
df_train.columns

# %%
df_train

# %%
df_test.columns

# %%
features = [ 'age', 'fnlwgt', 'educational_num', 'capital_gain',
       'capital_loss', 'hours_per_week', 'profile_description',
       'workclass_Local-gov', 'workclass_Private', 'workclass_Self-emp-inc',
       'workclass_Self-emp-not-inc', 'workclass_State-gov',
       'workclass_Without-pay', 'education_11th', 'education_12th',
       'education_1st-4th', 'education_5th-6th', 'education_7th-8th',
       'education_9th', 'education_Assoc-acdm', 'education_Assoc-voc',
       'education_Bachelors', 'education_Doctorate', 'education_HS-grad',
       'education_Masters', 'education_Preschool', 'education_Prof-school',
       'education_Some-college', 'marital_status_Married-AF-spouse',
       'marital_status_Married-civ-spouse',
       'marital_status_Married-spouse-absent', 'marital_status_Never-married',
       'marital_status_Separated', 'marital_status_Widowed',
       'occupation_Armed-Forces', 'occupation_Craft-repair',
       'occupation_Exec-managerial', 'occupation_Farming-fishing',
       'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
       'occupation_Other-service', 'occupation_Priv-house-serv',
       'occupation_Prof-specialty', 'occupation_Protective-serv',
       'occupation_Sales', 'occupation_Tech-support',
       'occupation_Transport-moving', 'gender_Male', 'native_country_Canada',
       'native_country_China', 'native_country_Columbia',
       'native_country_Cuba', 'native_country_Dominican-Republic',
       'native_country_Ecuador', 'native_country_El-Salvador',
       'native_country_England', 'native_country_France',
       'native_country_Germany', 'native_country_Greece',
       'native_country_Guatemala', 'native_country_Haiti',
       'native_country_Honduras', 'native_country_Hong',
       'native_country_Hungary', 'native_country_India', 'native_country_Iran',
       'native_country_Ireland', 'native_country_Italy',
       'native_country_Jamaica', 'native_country_Japan', 'native_country_Laos',
       'native_country_Mexico', 'native_country_Nicaragua',
       'native_country_Outlying-US(Guam-USVI-etc)', 'native_country_Peru',
       'native_country_Philippines', 'native_country_Poland',
       'native_country_Portugal', 'native_country_Puerto-Rico',
       'native_country_Scotland', 'native_country_South',
       'native_country_Taiwan', 'native_country_Thailand',
       'native_country_Trinadad&Tobago', 'native_country_United-States',
       'native_country_Vietnam', 'native_country_Yugoslavia']

x = df_train[features]
y = df_train['income']

x_final = df_test[features]

# %%
x.info()

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=42)

# %%
model1 = CatBoostClassifier(iterations=1000, learning_rate=0.1, verbose=100, cat_features=['profile_description'])

# %%
model1.fit(x_train, y_train)

# %%
y_pred = model1.predict(x_test)

# %%
y_pred

# %%
y_test

# %%
scor = f1_score(y_pred, y_test, pos_label=">50K")

# %%
scor

# %%
y_final = model1.predict(x_final)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


# %% [markdown]
# Vectorizare TF - IDF

# %%
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(df_train['profile_description'])

# %% [markdown]
# ELBOW method
# 
# 

# %%
wcss = []


# %%
for k in range(1,10):
    cluster = KMeans(n_clusters=k, random_state=42)
    cluster.fit(X1)
    wcss.append(cluster.inertia_) 


# %%
wcss

# %%
plt.plot(range(1,10))
plt.xlabel('clustere k')
plt.ylabel('wcss')
plt.show()

# %%
df_test['profile_description'].unique()

# %%
cluster = KMeans(n_clusters=4, random_state=42)

# %%
cluster.fit(X1)

# %%
x2 = vectorizer.fit_transform(df_test['profile_description'])

# %%
sub4 = cluster.predict(x2)

# %%
sub4

# %%
task1 = pd.DataFrame({
    'subtaskID': [1],
    'datapointID': [1],
    'answer': sub1
})

# %%
task2 = pd.DataFrame({
    'subtaskID': [2],
    'datapointID': [2],
    'answer': sub2
})

# %%
task3= pd.DataFrame({
    'subtaskID': 3,
    'datapointID': df_test['sampleid'],
    'answer': y_final
})

# %%
task4= pd.DataFrame({
    'subtaskID': 4,
    'datapointID': df_test['sampleid'],
    'answer': sub4
})

# %%
final = pd.concat([task1,task2,task3,task4], ignore_index=True)

# %%
final.to_csv('submission.csv', index=False)

# %%


# %%



