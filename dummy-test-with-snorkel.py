import os

if os.path.exists('data'):
    os.remove('data')

os.symlink('/opt/intents_labelling/data', 'data')

from intents_labelling.snorkel_labelling.snorkel_labelling import SnorkelLabelling
import pandas as pd

df = pd.DataFrame([{'qid': '1', 'query': 'facebook log in', 'url': ''}])
df = SnorkelLabelling().predict_first_level(df=df)

print(df)

df = SnorkelLabelling().predict_second_level(df=df)
print(df)