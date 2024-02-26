FROM webis/wows-2024-baseline:0.0.1-snorkel

RUN pip3 install fasttext~=0.9.2 gensim~=4.1.2 Levenshtein~=0.16.0 scikit-learn ir_datasets \
    && python3 -m spacy download en_core_web_lg \
    && rm -Rf /root/.cache

ADD snorkel_intent_predictor.py /code/snorkel_intent_predictor.py

ENTRYPOINT [ "python3", "/code/snorkel_intent_predictor.py" ]
