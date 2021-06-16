import random
import spacy
from spacy.training import Example
from spacy.util import minibatch
from spacy.util import compounding
import time

def listToTupleList(inputlist):
    l1 = []
    l2 = []
    for i in inputlist:
        l1.append(i.strip())
        t = tuple(l1)
        l2.append(t)
        l1 = []

    return '[' + ', '.join('({})'.format(t[0]) for t in l2) + ']'

def listT(inputlist):
    dd=[]
    for test_strr in inputlist:
        test_str=test_strr.strip()
        res = tuple(map(int, test_str.split(', ')))
        dd.append(res)
    return dd
def readTrainData():
    fp = open('train.txt', "r")
    lines = fp.readlines()
    print('-show read data in list-')
    print(lines)
    for line in lines:
        print(line)
    ret = []
    #ret = listToTupleList(lines)
    ret = listT(lines)
    return ret
LABEL = 'ANIMAL'
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
    ("我就是恋爱了", {'entities': [(3, 5, "emotion")]}),
    ("这听起来很开心", {'entities': [(5, 7, "emotion")]}),
    ("我目前是30岁", {'entities': [(4, 7, "age")]}),
    ("我目前是3岁", {'entities': [(4, 6, "age")]}),
    ("我朋友在台积电工作", {'entities': [(4, 7, "company")]}),
    ("高手", {'entities': [(0, 2, "skill")]}),
    ("1997年是我生日", {'entities': [(0, 5, "year")]}),
    ("我真的恋爱了", {'entities': [(3, 5, "emotion")]}),
    ("我没有恋爱", {'entities': [(3, 5, "emotion")]}),
    ("我想恋爱", {'entities': [(2, 4, "emotion")]}),
    ("我开心因为妳欢喜", {'entities': [(1, 3, "emotion")]}),
    ("我是30岁", {'entities': [(2, 5, "age")]}),
    ("我是20岁", {'entities': [(2, 5, "age")]}),
    ("我目前是1岁", {'entities': [(4, 6, "age")]}),
    ("我朋友在鸿海工作", {'entities': [(4, 6, "company")]}),
    ("我朋友在联电工作", {'entities': [(4, 6, "company")]}),
    ("我朋友在台积电工作", {'entities': [(4, 7, "company")]}),
    ("我朋友是高手很强", {'entities': [(4, 6, "skill")]}),
    ("我生于1997年", {'entities': [(3, 8, "year")]}),
    ("我出生于1997年的时后", {'entities': [(4, 9, "year")]}),
    ("我生于2003年", {'entities': [(3, 8, "year")]}),
    ("这三小东西?", {'entities': [(1, 3, "dirty")]})
]
"""
TRAIN_DATA = [
    ("我就是恋爱了", {'entities': [(3, 5, "emotion")]}),
    ("这听起来很开心", {'entities': [(5, 7, "emotion")]}),
    ("我真的恋爱了", {'entities': [(3, 5, "emotion")]}),
    ("我没有恋爱", {'entities': [(3, 5, "emotion")]}),
    ("我想恋爱", {'entities': [(2, 4, "emotion")]}),
    ("我开心因为妳欢喜", {'entities': [(1, 3, "emotion")]}),
]

#TRAIN_DATA1=readTrainData()
print('------ origin -------')
print(TRAIN_DATA)
print('-------load from file-------')
#print(TRAIN_DATA1)
#time.sleep(10)
#time.sleep(10)
#TRAIN_DATA=TRAIN_DATA1
for t in TRAIN_DATA:
    print(t)

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
iternum=20
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


