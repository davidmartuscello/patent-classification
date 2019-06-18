# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from numpy.random import rand

from torchtext.data import Field, LabelField, Dataset, Example, BucketIterator
import pandas as pd
import json

class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
         examples pd.DataFrame: DataFrame of examples
         fields {str: Field}: The Fields to use in this tuple. The
             string is a field name, and the Field is the associated field.
         filter_pred (callable or None): use only exanples for which
             filter_pred(example) is true, or use all examples if None.
             Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()

        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex
        # for key, tuple in fields.items():
        #     (name, field) = tuple
        #     if key not in data:
        #         raise ValueError("Specified key {} was not found in "
        #         "the input data".format(key))
        #     if field is not None:
        #         setattr(ex, name, field.preprocess(data[key]))
        #     else:
        #         setattr(ex, name, data[key])
        # return ex

def load_dataset(batch_size, cache_data=True, test_sen=None):

    if cache_data:
        print("Caching Data")
        office_actions = pd.read_csv('../data/office_actions.csv',
            index_col='app_id',
            usecols=['app_id', 'rejection_102', 'rejection_103'],
            dtype={'app_id':int, 'rejection_102': int, 'rejection_103': int},
            nrows=200000)

        abstractList = []
        idList = []
        rejectionColumn = []
        obviousCount = 0
        notCount = 0
        path = "/scratch/dm4350/json_files/"
        count = 0

        for filename in os.listdir(path):

            if count % 1000 == 0:
                print(count)

            filepath = path + filename
            try:
                jfile = open(filepath, 'r')
            except FileNotFoundError:
                print("File Not Found")
                continue

            try:
                parsed_json = json.load(jfile)
                jfile.close()
            except UnicodeDecodeError:
                print("WARNING: UnicodeDecodeError")
                continue
            except json.decoder.JSONDecodeError:
                print("WARNING: JSONDecodeError")
                continue

            app_id = int(filename.replace("oa_", "").replace(".json", "").replace("(1)", ""))
            try:
                row  = office_actions.loc[app_id]
            except KeyError:
                print("WARNING: KeyError")
                continue

            try:
                n = int(row.rejection_102)
                o = int(row.rejection_103)
            except TypeError:
                n = int(row.rejection_102.iloc[0])
                o = int(row.rejection_103.iloc[0])


            if n == 0 and o == 0:
                rejType = 0 #neither
            elif n == 0 and o == 1:
                rejType = 1 #obvious
            elif n == 1 and o == 0:
                rejType = 0 #novelty
            elif n == 1 and o == 1:
                rejType = 1 #both
            else:
                print("Office actions dataframe error:", sys.exc_info()[0])
                raise

            if obviousCount >= notCount and rejType == 1:
                continue

            obviousCount += o
            notCount += not(o)

            # Skip any files not in the appropriate IPC class
            try:
                for s in parsed_json[0]['ipc_classes']:
                    if (s.find("A61") != -1):
                        break
                    continue
            except:
                print("WARNING: file "+filepath+" is empty!\n")
                continue

            # Read in data from json file if it exists
            try:
                a = parsed_json[0]['abstract_full']
                i = parsed_json[0]['application_number']
            except IndexError:
                print("WARNING: file "+filepath+" is empty!\n")
                continue
            except KeyError:
                print("WARNING: file "+filepath+" is empty!\n")
                continue


            abstractList.append(a)
            idList.append(i)
            rejectionColumn.append(rejType)

            count += 1
            #if count > 2000: break

        df = pd.DataFrame({'text':abstractList, 'label':rejectionColumn}, index = idList)
        print("{} files loaded".format(count))

        df.to_pickle('./data_cache/abstracts_df_A61.pkl')
        # with open("data_cache/TEXT.Field","wb")as f:
        #     dill.dump(TEXT,f)
        # with open("data_cache/LABEL.Field","wb")as f:
        #     dill.dump(LABEL,f)

    else:
        print('Loading Dataset from Cache')
        df = pd.read_pickle('./data_cache/abstracts_df_A61.pkl')
        # with open("data_cache/TEXT.Field","rb")as f:
        #     TEXT=dill.load(f)
        # with open("data_cache/LABEL.Field","rb")as f:
        #     LABEL=dill.load(f)

    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = LabelField(sequential=False)

    fields={'text': TEXT, 'label': LABEL}
    ds = DataFrameDataset(df, fields)

    TEXT.build_vocab(ds, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(ds)

    train_data, test_data = ds.split()
    train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_iter, valid_iter, test_iter = BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
