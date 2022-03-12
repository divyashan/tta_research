

import os
os.environ['TRANSFORMERS_CACHE'] = '/local/helenl/.cache/'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/local/helenl/.cache/'

import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from tqdm.notebook import tqdm

from gpu_utils import restrict_GPU_pytorch
from helenl_utils import *

restrict_GPU_pytorch('0')


# Load Amazon WILDS pre-trained model

import statistics
import sys
import pickle
import numpy as np
import pandas as pd
import statistics


import torch
import torchvision.transforms as transforms

import argparse
import pdb


from wordcloud import WordCloud




sentence_augmentations = [#'nlp_random_sentence_shuffle_left'
                          #, 'nlp_random_sentence_shuffle_right'
                          #, 'nlp_random_sentence_shuffle_neighbor'
                          #, 'nlp_random_sentence_shuffle_random'
                          #'nlp_contextual_sentence_insertion_gpt2_embedding'
                          #, 'nlp_contextual_sentence_insertion_xlnet_cased_embedding'
                          #, 'nlp_contextual_sentence_insertion_distilgpt2_embedding'
                          'nlp_abstractive_summarization_bart_large_cnn'
                          , 'nlp_abstractive_summarization_t5_small'
                          , 'nlp_abstractive_summarization_t5_base'
                          , 'nlp_abstractive_summarization_t5_large'
]


full_dataset = get_dataset(dataset="civilcomments", download=False, root_dir = './wilds/data')


for sentence_aug in sentence_augmentations:
    print(sentence_aug)
    predict_augmented_labels(sentence_aug
                             , "civilcomments"
                             , "ERM"
                             , "./data/civilcomments_ERM_predictions_optimized_params/raw/"
                             , full_dataset = full_dataset
                            , num_samples = 1
                            , aug_sentence_min = 1
                            , aug_sentence_max = 1
                            , aug_sentence_p = 1)
    

for sentence_aug in sentence_augmentations:
    print(sentence_aug)
    predict_augmented_labels(sentence_aug
                             , "civilcomments"
                             , "ERM"
                             , "./data/civilcomments_ERM_predictions_optimized_params_four_samples/raw/"
                             , full_dataset = full_dataset
                            , num_samples = 4
                            , aug_sentence_min = 1
                            , aug_sentence_max = 1
                            , aug_sentence_p = 1)


