import random
import spacy
from spacy.training import Example
from spacy.util import minibatch
from spacy.util import compounding
import pickle

LABEL = 'emotion'
"""
TRAIN_DATA = [
    ("马有很长的尾巴", {'entities': [(0, 1, LABEL)]}),
    ("Do they bite?", {'entities': []}),
    ("马爱吃草", {'entities': [(0, 1, LABEL)]}),
    ("马比狗还大只", {'entities': [(0, 1, LABEL)]}),
    ("那些奔跑的动物就是马", {'entities': [(9, 10, LABEL)]}),
    ("那邊有馬跟狗", {'entities': [(3, 4, LABEL), (5,6,LABEL)]}),
    ("馬?", {'entities': [(0, 1, LABEL)]})
]
"""
"""
TRAIN_DATA = [
    ("李登辉的最后一本书", {'entities': [(0, 3, LABEL)]}),
    ("Do they bite?", {'entities': []}),
    ("美国同意台湾总统李登辉6月「私人访问」康乃尔大学", {'entities': [(8, 11, LABEL)]}),
    ("李登辉把民主自由留给台湾", {'entities': [(0, 3, LABEL)]}),
    ("李登辉执政十二年", {'entities': [(0, 3, LABEL)]}),
    ("李登辉预备前往康乃尔大学攻读农业经济博士", {'entities': [(0, 3, LABEL)]}),
    ("李登辉?", {'entities': [(0, 3, LABEL)]})
]
"""

with open ('./formating/train', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)
print(TRAIN_DATA)


#nlp = spacy.load('en_core_web_sm')  # load existing spaCy model
#ner = nlp.get_pipe('ner')
nlp = spacy.blank("zh")
ner = nlp.create_pipe("ner")
nlp.add_pipe('ner')
#nlp.add_pipe(ner, last=True)
ner=nlp.get_pipe("ner")
ner.add_label(LABEL)
#print(ner.move_names) # Here I see, that the new label was added
#optimizer = nlp.create_optimizer()
optimizer = nlp.begin_training()
# get names of other pipes to disable them during training
#other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
iternum=22
#iternum=60
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
output_dir='./zhtrainanimal.train'
#nlp.tokenizer = 'jieba'
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
test_text = '当你住在美国时，你喜欢马跟鸟'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)

test_text = '李登辉的最后一本书'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)

test_text = '李子強的最后一本书'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)

test_text = '李子强明天要去上课'
doc = nlp(test_text)
print("Entities in '%s'" % test_text)
for ent in doc.ents:
    print(ent.label_, " -- ", ent.text)
