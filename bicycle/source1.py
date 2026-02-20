# %%
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# %%
df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_eval.csv')

# %%
df_train.head()

# %%
df_train.isnull().sum()

# %%
df_train['Month'] = df_train['Activity Date'].str[:3]

# %%
df_train['Month']

# %%
df_train['Speed'] = df_train['Distance'] / (df_train['Moving Time'] / 3600)

# %%
df_train['Speed']

# %%
monthly = df_train.groupby('Month').agg({
    'Distance': 'sum',
    'Moving Time': 'sum'
}).reset_index()


# %%
monthly

# %%
monthly['AvgSpeed'] = monthly['Distance'] / (monthly['Moving Time'] / 3600)

# %%
monthly

# %%
order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

monthly['Month'] = pd.Categorical(monthly['Month'], order)
monthly = monthly.sort_values('Month')

# %%
monthly['AvgSpeed'] = np.floor(monthly['AvgSpeed'] * 100000) / 10000

# %%
monthly

# %%
df_test

# %%
features = ['Distance', 'Elapsed Time'	,'Moving Time',	'Starting Latitude',	'Starting Longitude',	'Finish Latitude',	'Finish Longitude' ]
X = df_train[features]
y = df_train['Label']

X_eval = df_test[features]

# %%
y

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_eval_scaled = scaler.transform(X_eval)


# %%
X_scaled

# %%
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_scaled, y)

# %%
y_pred = model.predict(X_eval_scaled)

# %%
y_pred
df_test['Label'] = y_pred

# %%
sub1 = pd.DataFrame({
    'subtaskID': 1,
    'Answer1': monthly['Month'],
    'Answer2': monthly['AvgSpeed']
})

# %%
sub2 = pd.DataFrame({
    'subtaskID': 2, 
    'Answer1': df_test['Activity ID'],
    'Answer2': df_test['Label']
})

# %%
sub2

# %%
final = pd.concat([sub1, sub2], ignore_index=True)
final.to_csv('submission.csv', index=False)

# %%



