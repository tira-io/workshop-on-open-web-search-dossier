#see https://github.com/OpenWebSearch/wows-code/tree/main/ecir24/baselines/dev-container
FROM webis/wows-2024-baseline:0.0.5

RUN pip3 install git+https://github.com/ProjectDossier/intents_labelling.git

ADD snorkel_intent_predictor.py /code/snorkel_intent_predictor.py

ENTRYPOINT [ "python3", "/code/snorkel_intent_predictor.py" ]
