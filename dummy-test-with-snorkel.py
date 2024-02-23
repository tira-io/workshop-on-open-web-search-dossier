from intents_labelling.snorkel_labelling.snorkel_labelling import SnorkelLabelling
import pandas as pd

df = pd.DataFrame([{'qid': '1', 'query': 'facebook', 'url': ''}, {'qid': '1', 'query': 'buy playstation', 'url': ''}, {'qid': '1', 'query': 'download playstation', 'url': ''},  {'qid': '1', 'query': 'what salary cs', 'url': ''},  {'qid': '1', 'query': 'how to cheat in exam', 'url': ''}, {'qid': '1', 'query': 'mail', 'url': ''}])

df = SnorkelLabelling().predict_first_level(df=df)

print(df)

df = SnorkelLabelling().predict_second_level(df=df)
print(df)