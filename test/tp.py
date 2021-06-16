import spacy

aa=['bear','dog','elephant','cat']
animals = ('bear', 'cat', 'dog', 'elephant', 'cat')
print(animals)

d=()
for a in aa:
    print(((a)))
    print(tuple((a)))
s='12345'
print(tuple([s]))

ss=['123','456','789']
b=tuple(ss)
print(b)
print([b])

def listToTupleList1(ss):
    b=tuple(ss)
    #print(b)
    #print([b])
    return ([b])

def listToTupleList2(ss):
    d = []
    for s in ss:
        b=tuple(s)
        d.append(b)
    #print(b)
    #print([b])
    #return ([b])
    return d
def listToTupleList3(inputlist):
    #output = [tuple(j[1] for j in i) for i in inputlist]
    #output = [tuple(j[1] for j in i) for i in inputlist]
    #output = [(i, ) for i in inputlist]
    output = [tuple([value]) for value in inputlist]

def listToTupleList5(inputlist):
    l1 = []
    l2 = []
    for i in inputlist:
        l1.append(i.strip())
        t = tuple(l1)
        l2.append(t)
        l1 = []
    return '[' + ', '.join('({})'.format(t[0]) for t in l2) + ']'

def listToTupleList(inputlist):
    #return list(zip(inputlist))
    return list(list(zip(*inputlist)))

    #return output



ret = listToTupleList(ss)
print(ret)
s="""
[('123'),('234'),('567')]
"""
print('should be')
print(s)
for i in ret:
    print(i)
# should be [('123'),('234'),('567')]

b=()
a='asdfa'
b=(tuple(a))
print(b)

TRAIN_DATA = [
    ("Horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, 'LABEL')]}),
    ("Do they bite?", {'entities': []}),
    ("horses are too tall and they pretend to care about your feelings", {'entities': [(0, 6, 'LABEL')]}),
    ("horses pretend to care about your feelings", {'entities': [(0, 6, 'LABEL')]}),
    ("they pretend to care about your feelings, those horses", {'entities': [(48, 54, 'LABEL')]}),
    ("they pretend to care about your feelings, those horses and dogs", {'entities': [(48, 54, 'LABEL'), (59,63,'LABEL')]}),
    ("horses?", {'entities': [(0, 6, 'LABEL')]})
]

for t in TRAIN_DATA:
    print(type(t))
    print(t)


#testT=('asdfads')
def listT(inputlist):
    dd=[]
    for test_str in inputlist:
        res = tuple(map(int, test_str.split(', ')))
        dd.append(res)
    return dd


testT=tuple(str('asdfads'))
print(type(testT))
print(testT)
test_str='123456'
res = tuple(map(int, test_str.split(', ')))
print(res)
aa=['123','456','678']
print(listT(aa))
