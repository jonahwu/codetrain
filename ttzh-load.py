import random
import spacy
from spacy.training import Example
import time
from spacy.util import minibatch
from spacy.util import compounding


LABEL = 'AAANIMAL'
"""
TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("Do they bite?", {'entities': []}),
    ("horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("horses pretend to care about your feelings", {'entities': [(0, 6, LABEL)]}),
    ("they pretend to care about your feelings, those horses", {'entities': [(48, 54, LABEL)]}),
    ("they pretend to care about your feelings, those horses and dogs", {'entities': [(48, 54, LABEL), (59,63,LABEL)]}),
    ("horses?", {'entities': [(0, 6, LABEL)]})
]
"""
TRAIN_DATA = [
    ("马有很长的尾巴", {'entities': [(0, 1, LABEL)]}),
    ("马喜欢吃草", {'entities': [(0, 1, LABEL)]}),
    ("那些奔跑的动物是马", {'entities': [(8, 9, LABEL)]}),
    ("那边有马跟狗", {'entities': [(3, 4, LABEL), (5,6,LABEL)]}),
    ("马?", {'entities': [(0, 1, LABEL)]})
]
for t in TRAIN_DATA:
    print(t[0],len(t[0]))
#time.sleep(10)
#("马比狗还大只", {'entities': [(0, 1, LABEL)]}),
#nlp = spacy.load('en_core_web_sm')  # load existing spaCy model
nlp = spacy.load('zh_core_web_md')  # load existing spaCy model
ner = nlp.get_pipe('ner')
#nlp = spacy.blank("en")
#ner = nlp.create_pipe("ner")
#nlp.add_pipe(ner, first=True)
ner.add_label(LABEL)
print(ner.move_names) # Here I see, that the new label was added
optimizer = nlp.create_optimizer()
# get names of other pipes to disable them during training
iternum=22
for itn in range(iternum):
    #random.shuffle(TRAIN_DATA)
    losses = {}
    batches=minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update(
                [example],
                drop=0.1,
                losses=losses,
            )
    print(losses)
# test the trained model # add some dummy sentences with many NERs
output_dir='./zhtrainanimal.load'
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

#test_text = 'Do you like horses?'
test_text = '你喜欢马吗'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)
## adding trainiing line horse and dogs will konows dogs it ANIMAL and iterate set to 40; will be scucess or it will not be recognized
# try standard label Canada which is been defined.
#test_text = 'Do you like horses and birds when you lived in Canada?'
#test_text = 'When you live in America, do you like horses and birds'
test_text = '当你住在美国，你喜欢马跟鸟'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)

