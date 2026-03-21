# %%
import pandas as pd
import numpy as np


df = pd.read_csv('dataset.csv')

# %%
df['Timestamp']

# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from datetime import datetime

df = pd.read_csv('dataset.csv')

def parse_ts(s):
    return datetime.strptime(s, 'Jan %d, %Y, %I:%M:%S %p')
df['Timestamp_DT'] = df['Timestamp'].apply(parse_ts)
df['seconds'] = df['Timestamp_DT'].dt.second + df['Timestamp_DT'].dt.minute*60

# %%
df

# %%
res1 = pd.DataFrame({
    'id': 'GLOBAL',
    'subtaskID': 'task1',
    'answer': len(df['Timestamp'].unique())
}, index=[0])
res2 = pd.DataFrame({
    'id': 'GLOBAL',
    'subtaskID': 'task2',
    'answer': 5
}, index=[0])


# %%

df['num_id'] = (range(len(df)))
df

# %%


# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
os.environ["OMP_NUM_THREADS"] = "3"
th=0.9
new_df =[]

moments = df['Timestamp'].unique()

kmeans = KMeans(n_clusters=5,random_state=42)

for moment in moments:

    snapshot = df[df['Timestamp'] == moment]
    nums = snapshot[['X','Y']]
    second = snapshot['seconds'].unique()[0]
    
    kmeans.fit(nums)
    
    centroids = kmeans.cluster_centers_
    score = silhouette_score(nums,kmeans.predict(nums))
    part_res = [[f'{moment}|{k}', f'{centroids[k][0]}|{centroids[k][1]}'] for k in range(5)]
    for i in range(5):
        if score > th:
            new_df.append({'X':centroids[i][0],'Y':centroids[i][1],'seconds':second,'train':1,'id':part_res[i][0],'answer':part_res[i][1],'drone_ID':i})
        else:
            new_df.append({'X':centroids[i][0],'Y':centroids[i][1],'seconds':second,'train':0,'id':part_res[i][0],'answer':part_res[i][1],'drone_ID':i})
        


# %%
new_df = pd.DataFrame(new_df)

# %%
new_df

# %%
from sklearn.linear_model import LinearRegression

train_X, train_Y = new_df.loc[new_df['train'] == 1, ['seconds', 'X', 'drone_ID']].copy(), new_df.loc[new_df['train'] == 1, ['seconds', 'Y', 'drone_ID']].copy()

# %%
lx = LinearRegression().fit(train_X[['seconds']], train_X['X'])
ly = LinearRegression().fit(train_Y[['seconds']], train_Y['Y'])

# %%
new_df['predX'] = lx.predict(new_df[['seconds']])
new_df['predY'] = ly.predict(new_df[['seconds']])

# %%
new_df

# %%
for_r4 = []
for i in range(5):
    tx, ty = train_X[train_X['drone_ID'] == i].copy(), train_Y[train_Y['drone_ID'] == i].copy()
    lx.fit(tx[['seconds']], tx['X'])
    ly.fit(ty[['seconds']], ty['Y'])

    new_df.loc[new_df['drone_ID'] == i, 'predX'] = lx.predict(new_df.loc[new_df['drone_ID'] == i, ['seconds']])
    new_df.loc[new_df['drone_ID'] == i, 'predY'] = ly.predict(new_df.loc[new_df['drone_ID'] == i, ['seconds']])

    for_r4.append(f'{lx.predict([[500]])[0]}|{ly.predict([[500]])[0]}')

# %%


# %%
predX = list(new_df['predX'])
predY = list(new_df['predY'])
res3 = pd.DataFrame({
    'id': new_df['id'],
    'subtaskID': 'task3',
    'answer': [f'{predX[i]}|{predY[i]}' for i in range(len(predX))]
})


# %%
res4 = pd.DataFrame({
    'id': [f'Jan 26, 2026, 12:08:20 AM|{i}'for i in range(5)],
    'subtaskID': 'task4',
    'answer': for_r4
})
res4

# %%
full = pd.concat([res1,res2,res3,res4])
full.to_csv('submission.csv')

# %%
plt.scatter(new_df['predX'],new_df['seconds'])


