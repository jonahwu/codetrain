import random
import spacy
from spacy.training import Example

LABEL = 'ANIMAL'
TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("Do they bite?", {'entities': []}),
    ("horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("horses pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("they pretend to care about your feelings, those horses", {'entities': [(48, 54, LABEL)]}),
    ("they pretend to care about your feelings, those horses and dogs", {'entities': [(48, 54, LABEL), (59,63,LABEL)]}),
    ("horses?", {'entities': [(0, 6, LABEL)]})
]
nlp = spacy.load('en_core_web_sm')  # load existing spaCy model
ner = nlp.get_pipe('ner')
#nlp = spacy.blank("en")
#ner = nlp.create_pipe("ner")
#nlp.add_pipe(ner, first=True)
ner.add_label(LABEL)
print(ner.move_names) # Here I see, that the new label was added
optimizer = nlp.create_optimizer()
# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(60):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            #doc = nlp(text)
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
        print(losses)
# test the trained model # add some dummy sentences with many NERs
output_dir='trainanimal.load'
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

test_text = 'Do you like horses?'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)
## adding trainiing line horse and dogs will konows dogs it ANIMAL and iterate set to 40; will be scucess or it will not be recognized
# try standard label Canada which is been defined.
#test_text = 'Do you like horses and birds when you lived in Canada?'
test_text = 'When you live in America, do you like horses and birds'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)

