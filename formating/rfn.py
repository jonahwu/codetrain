
import pickle
#with open ('Data/ner_corpus_260', 'rb') as fp:
with open ('train', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)
print(TRAIN_DATA)

