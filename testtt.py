import random
import spacy
from spacy.training import Example

#nlp = spacy.load('trainanimal.train')  # load existing spaCy model
nlp = spacy.load('trainanimal.load')  # load existing spaCy model
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

