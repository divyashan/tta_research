import time
import json
import fasttext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial.distance import cdist, cosine, euclidean
from scipy.stats import ttest_ind, ttest_1samp
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from gpu_utils import restrict_GPU_pytorch

import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

restrict_GPU_pytorch('3')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.cuda()

# Returns a list of predictions (prob of moderation) for a list of examples and a given model. 
# allExamples: list of strings 
# model: fastText model object
def getModelPredsHelper(allExamples, model):
    exampleList = [x.replace('\n', ' ') for x in allExamples]
    preds = model.predict(exampleList)
    preds_int = np.array([1 if 'positive' in p[0] else 0 for p in preds[0]])
    preds_prob = np.array([p[0] for p in preds[1]])
    probs = np.array([1 - p if preds_int[i] == 0 else p for (i,p) in enumerate(preds_prob)])
    probs = [np.round(p,3) for p in probs]
    return probs

# vec: query vector (list of floats)
# all_vecs: pool of all vectors from which neighbors are retrieved (list of list of floats)
# comments: comments corresponding to the vectors in all_vecs (list of strings)
# n: number of neighbors to return (int)
# return_idx: if true, returns indices of nearest neighbors instead of the actual comments (bool)
def getKNNFromVector(vec, all_vecs, comments, n=30, return_idx=False):
    dist_vec = cdist(vec, all_vecs, 'cosine')
    top_vec_idx = np.argsort(dist_vec[0])[1:n+1]
    if return_idx: return top_vec_idx
    top_comments = np.array(comments)[top_vec_idx]
    top_comments = [c for c in top_comments]
    return top_comments


DATA_FPATH = '../data/reddit/'

all_comments = pd.read_csv(DATA_FPATH + 'all_comments_df')

sub = 'funny'
comments = all_comments[all_comments.subreddit == sub].body.values
labels = all_comments[all_comments.subreddit == sub].moderated.values

t0 = time.time()
for i in range(len(comments)):
    tokenized_comments = tokenizer(list(comments[i]), padding=True, truncation=True, return_tensors="pt")['input_ids']
    tokenized_comments = tokenized_comments.cuda()
    model(tokenized_comments)['last_hidden_state'].cpu().detach().numpy()
    if i % 100 == 0:
        print(f'{i} of 34347')
t1 = time.time()
print(f'{t1 - t0} seconds')