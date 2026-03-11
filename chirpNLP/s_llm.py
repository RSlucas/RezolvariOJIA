# %%


# %%
# %%
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# %% 1. Citim datele
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# %% 2. Subtask 1: lungimea fiecărui chirp
sub1 = df_test['chirp'].apply(lambda x: len(str(x)))

# %% 3. Subtask 2: numărul de apariții ale caracterului #
def exista(text):
    if pd.isna(text):
        return 0
    return text.count("#")

df_train['sub2_#'] = df_train['chirp'].apply(exista)
df_test['sub2_#'] = df_test['chirp'].apply(exista)

sub2 = df_test['sub2_#']

# %% 4. Curățare text (lowercase + elimina caractere speciale)
def clean(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

df_train['chirp'] = df_train['chirp'].apply(clean)
df_test['chirp'] = df_test['chirp'].apply(clean)

# %% 5. Pregătim datele pentru model
features = ['chirp', 'sub2_#']
x = df_train[features]
y = df_train['label']
x_final = df_test[features]

# %% 6. Split pentru validare rapidă
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# %% 7. ColumnTransformer: text + numeric
preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english'), 'chirp'),
    ('num', 'passthrough', ['sub2_#'])
])

# %% 8. Pipeline LogisticRegression
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=500))
])

# %% 9. Antrenăm modelul
pipe.fit(x_train, y_train)

# %% 10. Validare rapidă: calcul AUC
y_pred_proba = pipe.predict_proba(x_test)[:, 1]
scor = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC pe validare:", scor)

# %% 11. Predicții pentru Subtask 3
y_final = pipe.predict_proba(x_final)[:, 1]

# %% 12. Creăm DataFrame pentru subtasks
t1 = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': df_test['id'],
    'answer': sub1
})

t2 = pd.DataFrame({
    'subtaskID': 2,
    'datapointID': df_test['id'],
    'answer': sub2
})

t3 = pd.DataFrame({
    'subtaskID': 3,
    'datapointID': df_test['id'],
    'answer': y_final
})

# %% 13. Concatenăm toate subtasks
final = pd.concat([t1, t2, t3], ignore_index=True)

# %% 14. Salvăm fișierul de submission
final.to_csv('submission_GPT.csv', index=False)

# %%
print("Submission gata! Primele 5 linii:")
print(final.head())

# %%



