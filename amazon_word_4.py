

import os
os.environ['TRANSFORMERS_CACHE'] = '/local/helenl/.cache/'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/local/helenl/.cache/'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
from tqdm.notebook import tqdm

from gpu_utils import restrict_GPU_pytorch
from helenl_utils import *

restrict_GPU_pytorch('3')


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




word_augmentations = [
                           'nlp_random_token_split'
                           , 'nlp_random_word_crop'
]


word_augmentations = [ # 'nlp_random_word_swap'
                      #, 'nlp_random_word_delete'
                      # , 'nlp_random_word_substitute'
                      'nlp_random_token_split'
                      , 'nlp_wordnet_synonym'
                      ,  'nlp_ppdb_synonym'
                       , 'nlp_antonym'
                       , 'nlp_random_contextual_word_insertion_bert_uncased_embedding'
                       , 'nlp_random_contextual_word_insertion_bert_cased_embedding'
                       , 'nlp_random_contextual_word_insertion_distilbert_uncased_embedding'
                       , 'nlp_random_contextual_word_insertion_distilbert_cased_embedding'
                       , 'nlp_random_contextual_word_insertion_roberta_base_embedding'
                       , 'nlp_random_contextual_word_insertion_distilroberta_base_embedding'
                       , 'nlp_random_contextual_word_insertion_bart_base_embedding'
                       , 'nlp_random_contextual_word_insertion_squeezebert_uncased_embedding'
                       , 'nlp_random_contextual_word_substitution_bert_uncased_embedding'
                       , 'nlp_random_contextual_word_substitution_bert_cased_embedding'
                       , 'nlp_random_contextual_word_substitution_distilbert_uncased_embedding'
                       , 'nlp_random_contextual_word_substitution_distilbert_cased_embedding'
                       , 'nlp_random_contextual_word_substitution_roberta_embedding'
                       , 'nlp_random_contextual_word_substitution_distilroberta_base_embedding'
                       , 'nlp_random_contextual_word_substitution_bart_base_embedding'
                       , 'nlp_random_contextual_word_substitution_squeezebert_uncased_embedding'
    
]


char_augmentations = ['bert'
                       , 'nlp_ocr'
                       , 'nlp_keyboard'
                       , 'nlp_random_char_insert'
                       , 'nlp_random_char_substitution'
                       , 'nlp_random_char_swap'
                       , 'nlp_random_char_deletion'
                       , 'nlp_spelling_substitution']

full_dataset = get_dataset(dataset="amazon", download=False, root_dir = './wilds/data')

"""
for char_aug in char_augmentations:
	predict_augmented_labels(char_aug
				, "amazon"
				, "ERM"
				, "/data/ddmg/prism/tta/outputs/amazon_ERM_predictions_optimized_params_four_samples/raw/"
				, full_dataset = full_dataset
				, num_samples = 4
				, aug_char_min = 1
				, aug_char_max = 1
				, aug_char_p = 1
				, aug_word_min = 1
				, aug_word_max = None
				, aug_word_p = 0.1
				)
"""
for word_aug in word_augmentations:
	predict_augmented_labels(word_aug
                         , "amazon"
                         , "ERM"
                         , "/data/ddmg/prism/tta/outputs/amazon_ERM_predictions_optimized_params_four_samples/raw/"
                         , full_dataset = full_dataset
                        , num_samples = 4
                        , aug_char_min = 1
                        , aug_char_max = 1
                        , aug_char_p = 1
                        , aug_word_min = 1
                        , aug_word_max = 1
                        , aug_word_p = 1
                        , min_char = 4)


