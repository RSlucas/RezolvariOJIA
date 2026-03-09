# %%
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# %%
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# %%
df_train.info()

# %%
sub1 = len(df_train)

# %%
sub1

# %%
sub2 = (df_train['Gender'] == 'male').sum()

# %%
sub2

# %%
sub3 = df_train['Duration'].mean()

# %%
sub3

# %%
sub4 = (df_train['Age'] >= 75).sum()

# %%
sub4

# %%
df_train= (pd.get_dummies(df_train, columns=['Gender'], drop_first=True)).astype(int)
df_test= (pd.get_dummies(df_test, columns=['Gender'], drop_first=True)).astype(int)

df_train

# %%
df_train.columns

# %%
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp', 'Gender_male']

x = df_train[features]
y = df_train['Calories']
x_final = df_test[features]

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=42, test_size=0.3)

# %%
model1 = XGBRegressor(
    n_estimators = 3000,
    learning_rate = 0.01,
    random_state = 42
)

# %%
model1

# %%
model1.fit(x_train, y_train)

# %%
y_pred = model1.predict(x_test)

# %%
scor = mean_absolute_error(y_pred, y_test)
print(scor)

# %%
model2 = CatBoostRegressor(
    iterations=3000,
    random_seed=42, 
    verbose=100,
    early_stopping_rounds=200,
    learning_rate=0.02
)

# %%
model2.fit(x_train,y_train)

# %%
y_pred2 = model2.predict(x_test)

# %%
y_pred2

# %%
scor2 = mean_absolute_error(y_pred2,y_test)
print(scor2)

# %%
sub5 = model2.predict(x_final)

# %%
sub5

# %%
df_trainsub6 = df_train[df_train['Gender_male'] == 1]
df_testsub6 = df_test[df_test['Gender_male'] == 1]

# %%
df_trainsub6

# %%
x_sub6 = df_trainsub6[features]
y_sub6 = df_trainsub6['Calories']

x_final6 = df_testsub6[features]

# %%
x_train6, x_test6, y_train6, y_test6 = train_test_split(x_sub6, y_sub6, random_state=42, test_size=0.3)

# %%
model_sub6 = CatBoostRegressor(
    iterations=3000,
    verbose=100,
    learning_rate=0.02,
    early_stopping_rounds=200,
    random_seed=42
)

# %%
model_sub6.fit(x_train6, y_train6)

# %%
sub6 = model_sub6.predict(x_final6)

# %%
sub6

# %%
t1 = pd.DataFrame({
    'subtaskID': [1],
    'datapoinID': [1],
    'answer': sub1
})

# %%
t2 = pd.DataFrame({
    'subtaskID': [2],
    'datapoinID': [2],
    'answer': sub2
})

# %%
t3 = pd.DataFrame({
    'subtaskID': [3],
    'datapoinID': [3],
    'answer': sub3
})

# %%
t4 = pd.DataFrame({
    'subtaskID': [4],
    'datapoinID': [4],
    'answer': sub4
})

# %%
t5 = pd.DataFrame({
    'subtaskID': 5,
    'datapoinID': df_test['User_ID'],
    'answer': sub5
})

# %%
t6 = pd.DataFrame({
    'subtaskID': 6,
    'datapoinID': df_testsub6['User_ID'],
    'answer': sub6
})

# %%
final = pd.concat([t1,t2,t3,t4,t5,t6], ignore_index=True)

# %%
final

# %%
final.to_csv('output.csv', index=False)

# %%



