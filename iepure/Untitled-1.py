# %%
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, mean_squared_error

from sklearn.preprocessing import StandardScaler


# %%
df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')

# %%
df_train.info()

# %%
frame_sub1 = df_test[(df_test['Sex'] == 'Femelă') & (df_test['Ureche_lăsată'] == True) & (df_test['Culoare'] == 'Havana')]

# %%
sub1 = len(frame_sub1)

# %%
sub1

# %%
df_train.columns

# %%
featuers_cluster = ['Greutate', 'Lungime_urechi', 'Ureche_lăsată', 'Culoare',
       'Vârstă', 'Tip_blană', 'Calitate_blană', 'Formă_corp', 'Apare_gușa',
       'Sănătate']
df_cluster = df_test[featuers_cluster]
df_train_cluster = df_train[featuers_cluster]

# %%
df_cluster.select_dtypes(include='object').columns.tolist

# %%
df_cluster['Ureche_lăsată'] = df_cluster['Ureche_lăsată'].map({True: 1, False:0})
df_train_cluster['Ureche_lăsată'] = df_train_cluster['Ureche_lăsată'].map({True: 1, False:0})

# %%
df_cluster = pd.get_dummies(df_cluster, columns=['Culoare', 'Tip_blană', 'Calitate_blană', 'Formă_corp', 'Sănătate'], drop_first=False)
df_train_cluster = pd.get_dummies(df_train_cluster, columns=['Culoare', 'Tip_blană', 'Calitate_blană', 'Formă_corp', 'Sănătate'], drop_first=False)

# %%
df_cluster.info()

# %%
for cols in df_cluster.select_dtypes(include='bool').columns:
    df_cluster[cols] = df_cluster[cols].astype(int)

for cols in df_train_cluster.select_dtypes(include='bool').columns:
    df_train_cluster[cols] = df_train_cluster[cols].astype(int)

# %%
df_cluster.info()

# %%
x_cluster = df_cluster
x_train_cluster = df_train_cluster

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train_cluster)
x_scaled_final = scaler.transform(x_cluster)

# %%
model_cluster = KMeans(n_clusters=3, random_state=42, max_iter=500, n_init=20)

# %%
model_cluster.fit(x_scaled)

# %%
sub2 = model_cluster.predict(x_scaled_final)

# %%
sub2

# %%
df_train.columns

# %%
catego = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()
features = [ 'Sex', 'Greutate', 'Lungime_urechi', 'Ureche_lăsată', 'Culoare',
       'Vârstă', 'Tip_blană', 'Calitate_blană', 'Formă_corp', 'Apare_gușa',
       'Sănătate']

x = df_train[features]
y = df_train['Scor_jurizare']

x_final = df_test[features]

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)

# %%
model1 = CatBoostRegressor(iterations=1000, learning_rate=0.1, verbose=100, cat_features=catego)

# %%
model1.fit(x_train,y_train)

# %%
y_pred = model1.predict(x_test)

# %%
y_pred

# %%
scor = mean_squared_error(y_pred, y_test)

# %%
scor

# %%
y_final = model1.predict(x_final)

# %%
y_final = (np.clip(y_final, 0, 100)).round(2)

# %%
y_final

# %%
task1 = pd.DataFrame({
    'subtaskID': [1],
    'datapointID': [1],
    'answer': sub1
})

# %%
task2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': df_test['ID'],
    'answer': sub2
})

# %%
task3 = pd.DataFrame({
    'subtaskID': 3,
    'datapointID': df_test['ID'],
    'answer': y_final
})

# %%
final = pd.concat([task1, task2, task3], ignore_index=True)

# %%
final.to_csv('output.csv', index=False)

# %%


# %%



