import plac
import logging
import argparse
import sys
import os
import json
import pickle
import time

@plac.annotations(input_file=("Input file", "option", "i", str), output_file=("Output file", "option", "o", str))

def main(input_file=None, output_file=None):
    try:
        training_data = []
        lines=[]
        with open(input_file) as f:
            #lines = f.readlines()
            lines = json.load(f)
            print(lines)
            for line in lines:
                print(line)
                c,s,e,lb=getAttribute(line)
                print(c,s,e,lb)
                #print(l[0])
            #time.sleep(10)

        for line in lines:
            c,s,e,lb=getAttribute(line)
            #data = json.loads(line)
            text = c
            entities = []

            entities.append((s, e ,lb))


            training_data.append((text, {"entities" : entities}))

        print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None
"""
def mainnew(input_file=None, output_file=None):
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None
"""

def getAttribute(l):
    c=l[0]
    s=l[1]['entities'][0][0]
    e=l[1]['entities'][0][1]
    lb=l[1]['entities'][0][2]
    #content, start, end, label
    return c, s, e, lb

if __name__ == '__main__':
    plac.call(main)
