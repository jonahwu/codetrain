
import spacy


#nlp1 = spacy.load(R"./content/ner_demo/training/model-best") #load the best model
nlp1=spacy.load("./zhtrainanimal.train") #load the best model
#nlp1=spacy.blank("zh").from_disk("./zhtrainanimal.train")
#nlp1.tokenizer.initialize(pkuseg_model="mixed")


def runCopus(text):
        #doc = nlp1("这是什么鬼三小啦") # input sample tex
        doc = nlp1(text) # input sample tex
        print([(ent.text, ent.label_) for ent in doc.ents])
def test():
    doc = nlp1("这是什么鬼三小啦") # input sample tex
    print([(ent.text, ent.label_) for ent in doc.ents])
    doc = nlp1("这是什么鬼三小拉拉事情啦") # input sample tex
    print([(ent.text, ent.label_) for ent in doc.ents])
    doc1 = nlp1("我是真的恋爱了") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我想永无止境的恋爱") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("恋爱") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我不想只是恋爱而已") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])

    doc1 = nlp1("我就只是想开心而已") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])

    doc1 = nlp1("我今年30岁") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我今年10岁") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我今年2岁") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我今年99岁") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我今年101岁") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我今年1001岁") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])

    doc1 = nlp1("我朋友在台泥工作") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我朋友在华新科工作") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("我朋友在台肥工作") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("他很厉害绝对是高手") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("他真是高手界中的翘楚") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])
    doc1 = nlp1("老师真高手也很长") # input sample tex
    print([(ent.text, ent.label_) for ent in doc1.ents])

def run():
    fp = open('./assets/inf.txt', "r")
    line1 = fp.readlines()
    fp = open('testcopus.txt', "r")
    line2 = fp.readlines()
    lines=line1+line2
    for line in lines:
        if len(line)>1:
            print('---------------------------------')
            print(line)
            runCopus(line)


#test()
run()
