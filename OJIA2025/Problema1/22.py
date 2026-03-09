# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor 

# %%
df_train = pd.read_csv('train_data.csv')
df_test= pd.read_csv('test_data.csv')

# %%
df_train

# %%
samples = len(df_train)

# %%
subtask1 = samples

# %%
x=0
for om in df_train['Gender']:
    if om == "male":
        x=x+1

# %%
subtask2 = x

# %%
averageduration = df_train['Duration'].mean()

# %%
subtask3 = averageduration

# %%
y=0
for ani in df_train['Age']:
    if ani >= 75:
        y=y+1


# %%
subtask4 = y

# %%
subtask4

# %%
df_train.columns

# %%
df_test.info()

# %%
#def ids(value):
    #for om in df_train['Gender']:
       # if om == "male":
           # df_test['idsMALE'] = value

# %%
#df_train['User_ID'].apply(ids)

# %%
df_train = pd.get_dummies(df_train, columns =['Gender'], dtype=int, drop_first=False, prefix='gender')
df_test = pd.get_dummies(df_test, columns =['Gender'], dtype=int, drop_first=False, prefix='gender')


# %%
df_train.columns

# %%
df_train

# %%
#sns.pairplot(df_train)
#plt.show()

# %%
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp','gender_male', 'gender_female' ]
X = df_train[features]
X_eval = df_test[features]

# %%
target = 'Calories'
y = df_train[target]

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_final = scaler.transform(X_eval)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# %%
model_2 = LGBMRegressor(n_estimators=350, max_depth=15, random_state=42)
model_2.fit(X_train, y_train)

# %%
y_model_2 = model_2.predict(X_test)

# %%
mae1 = mean_absolute_error(y_test, y_model_2)
print(mae1)

# %%
y_finalll = model_2.predict(X_final)

# %%
rez = pd.DataFrame({'subtaskID' : 1, 'datapointID' : [1], 'answer' : subtask1})
rez1 =pd.DataFrame({'subtaskID' : 2, 'datapointID' : [1], 'answer' : subtask2})
rez2 = pd.DataFrame({'subtaskID' : 3, 'datapointID' : [1], 'answer' : subtask3})
rez3 = pd.DataFrame({'subtaskID' : 4,'datapointID' : [1], 'answer' : subtask4})
rez4 = pd.DataFrame({'subtaskID' : 5, 'datapointID' : df_test['User_ID'], 'answer' : y_finalll })


# %%
final = pd.concat([rez, rez1, rez2, rez3, rez4], ignore_index=True)

# %%
final

# %%
final.to_csv('submission.csv', index=False)

# %%



