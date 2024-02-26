#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import ir_datasets, get_input_directory_and_output_directory
from pathlib import Path
import pandas as pd
import gzip
import json

import os

if os.path.exists('data'):
    os.remove('data')

os.symlink('/opt/intents_labelling/data', 'data')

from intents_labelling.snorkel_labelling.snorkel_labelling import SnorkelLabelling

def process_query(query):
    df = pd.DataFrame([{'qid': '1', 'query': query['query'], 'url': query['url']}])
    df = SnorkelLabelling().predict_first_level(df=df)
    df = SnorkelLabelling().predict_second_level(df=df)
    df = SnorkelLabelling().create_final_label(df=df)
    return {'qid': query['qid'], 'intent_prediction': df.iloc[0]["Label"]}

def process_queries(queries_iter):
    return pd.DataFrame([process_query(i) for i in queries_iter])

def load_query_to_url_of_top_document(input_dir):
    ret = []
    with gzip.open(input_dir + '/rerank.jsonl.gz') as f:
        for i in f:
            i = json.loads(i)
            ret += [{'qid': i['qid'], 'query': i['query'], 'score': i['score'], 'url': i['original_document']['url']}]
        
        
    ret = pd.DataFrame(ret)
    ret = ret.sort_values(["qid", "score"], ascending=[True,False]).reset_index().groupby("qid").head(1)
    return [i[1] for i in ret.iterrows()]

if __name__ == '__main__':
    # In the TIRA sandbox, this is the injected ir_dataset, injected via the environment variable TIRA_INPUT_DIRECTORY
    input_dir, output_dir = get_input_directory_and_output_directory('post-retrieval-query-intent-predictor/tiny-input-data')

    
    # Query processors persist their results in a file queries.jsonl in the output directory.
    output_file = Path(output_dir) / 'queries.jsonl'
    
    # You can pass as many additional arguments to your program, e.g., via argparse, to modify the behaviour
    
    # process the queries, store results at expected location.
    queries = load_query_to_url_of_top_document(input_dir)
    processed_queries = process_queries(queries)
    print(output_dir)
    processed_queries.to_json(output_dir + '/queries.jsonl', lines=True, orient='records')
