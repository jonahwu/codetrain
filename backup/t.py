import spacy
import random
import json
from spacy.training.example import Example

nlp = spacy.blank("en")

TRAINING_DATA=[('Part List', {'entities': []}), ('pending', {'entities': []}), ('3D Printing', {'entities': [(0, 11, 'Process')]}), ('Recommended to use a FDM 3D printer with PLA material.', {'entities': [(25, 36, 'Process'), (41, 44, 'Material')]}), ('ï»¿', {'entities': []}), ('No need supports or rafts.', {'entities': []}), ('Resolution: 0.20mm', {'entities': []}), ('Fill density 20%', {'entities': []}), ('As follows from the analysis, part of the project is devoted to 3D', {'entities': [(64, 66, 'Process')]}), ('printing, as all static components were created using 3D modelling and', {'entities': [(54, 66, 'Process')]}), ('subsequent printing.', {'entities': []}), ('ï»¿', {'entities': []}), ('In our project, we created several versions of the', {'entities': []}), ('model during modelling, which we will describe and document in the', {'entities': []}), ('following subchapters. As a tool for 3D modelling, we used the Sketchup', {'entities': [(37, 49, 'Process')]}), ('Make tool, version from 2017. The main reason was the high degree of', {'entities': []}), ('intuitiveness and simplicity of the tool, as we had not encountered 3D', {'entities': [(68, 70, 'Process')]}), ('modelling before and needed a relatively flexible and efficient tool to', {'entities': []}), ('guarantee the desired result. with zero previous experience.', {'entities': []}), ('In this version, which is shown in the figures Figure 13 - Version no. 2 side view and Figure 24 - Version no. 2 - front view, for the first time, the specific dimensions of the infuser were clarified and', {'entities': []}), ('modelled. The details of the lower servo attachment, the cable hole in', {'entities': []}), ('the main mast, the winding cylinder mounting, the protrusion on the', {'entities': [(36, 44, 'Process')]}), ('winding cylinder for holding the tea bag, the preparation for fitting', {'entities': []}), ('the wooden and aluminium plate and the shape of the cylinder end that', {'entities': [(15, 25, 'Material')]}), ('exactly fit the servo were also reworked.', {'entities': []}), ('After the creation of this', {'entities': []}), ('version of the model, this model was subsequently officially consulted', {'entities': []}), ('and commented on for the first time.', {'entities': []}), ('In this version, which is shown in the figures Figure 13 - Version no. 2 side view and Figure 24 - Version no. 2 - front view, for the first time, the specific dimensions of the infuser were clarified and', {'entities': []}), ('modelled. The details of the lower servo attachment, the cable hole in', {'entities': []}), ('the main mast, the winding cylinder mounting, the protrusion on the', {'entities': [(36, 44, 'Process')]})]

ner = nlp.create_pipe("ner")
nlp.add_pipe('ner')
ner.add_label("label")
# Start the training
nlp.begin_training()
# Loop for 40 iterations
losses = {}
for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
    for text, annotations in batch:
        # create Example
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        # Update the model
        nlp.update([text], losses=losses, drop=0.3)
"""

for itn in range(40):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}
# Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]
# Update the model
        nlp.update(texts, annotations, losses=losses, drop=0.3)
    print(losses)
"""
