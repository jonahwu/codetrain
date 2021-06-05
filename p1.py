import spacy
from spacy.gold import offsets_from_biluo_tags, GoldParse
from spacy.util import minibatch, compounding
import random
from tqdm import tqdm

def train_spacy(data, iterations, model=None):
    TRAIN_DATA = data
    print(f"downloads = {model}")
    if model is not None and path.exists(model):
        print(f"training existing model")
        nlp = spacy.load(model)
        print("Model is Loaded '%s'" % model)
    else:
        print(f"Creating new model")

        nlp = spacy.blank('en')  # create blank Language class

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    # Based on template, get labels and save those for further training
    LABEL = ["Name", "ORG"]

    for i in LABEL:
        # print(i)
        ner.add_label(i)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        if model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.entity.create_optimizer()
        tags = dict()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 16.0, 1.001))
            # type 2 with mini batch
            for batch in batches:
                texts, _ = zip(*batch)
                golds = [GoldParse(nlp.make_doc(t),entities = a) for t,a in batch]
                nlp.update(
                    texts,  # batch of texts
                    golds,  # batch of annotations
                    drop=0.4,  # dropout - make it harder to memorise data
                    losses=losses,
                    sgd=optimizer
                )
            print(losses)
    return nlp

data_biluo = [
    ('I am Shah Khan, I work in MS Co', ['O', 'O', 'B-Name', 'L-Name', 'O', 'O', 'O', 'O', 'B-ORG', 'L-ORG']),
    ('I am Tom Tomb, I work in Telecom Networks', ['O', 'O', 'B-Name', 'L-Name', 'O', 'O', 'O', 'O', 'B-ORG', 'L-ORG'])
]


model = train_spacy(data_biluo, 10)
