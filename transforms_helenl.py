import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import torch

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import os

model_dir = '../models/'


def initialize_transform(transform_name, config, dataset, is_training = True, num_aug_samples = 1, aug_size = 5):
    """
    By default, transforms should take in `x` and return `transformed_x`.
    For transforms that take in `(x, y)` and return `(transformed_x, transformed_y)`,
    set `do_transform_y` to True when initializing the WILDSSubset.    
    """

    if transform_name is None:
        return None
    elif transform_name=='bert':
        return initialize_bert_transform(config, num_aug_samples, aug_size)
    elif transform_name=='image_base':
        return initialize_image_base_transform(config, dataset)
    elif transform_name=='image_resize_and_center_crop':
        return initialize_image_resize_and_center_crop_transform(config, dataset)
    elif transform_name=='poverty':
        return initialize_poverty_transform(is_training)
    elif transform_name=='rxrx1':
        return initialize_rxrx1_transform(is_training)
    elif 'nlp' in transform_name:
        print(aug_size)
        return initialize_bert_transform(config, aug_size)
    else:
        raise ValueError(f"{transform_name} not recognized")

        
        
def initialize_nlpaug_transform(transform_name, aug_size):
    
    if transform_name == 'nlp_ocr':
        aug = nac.OcrAug()
    
    elif transform_name == 'nlp_keyboard':
        aug = nac.KeyboardAug()
        
    elif transform_name == 'nlp_random_char_insert':
        aug = nac.RandomCharAug(action = 'insert')
    
    elif transform_name == 'nlp_random_char_substitution':
        aug = nac.RandomCharAug(action = 'insert')
        
    elif transform_name == 'nlp_random_char_swap':
        aug = nac.RandomCharAug(action = 'swap')
        
    elif transform_name == 'nlp_random_char_deletion':
        aug = nac.RandomCharAug(action = 'delete'
                               , aug_word_p = 0.20
                               , aug_char_min = aug_size
                               , aug_char_max = aug_size
                               , aug_char_p = 1)
        
    
    elif transform_name == 'nlp_spelling_substitution':
        aug = naw.SpellingAug(aug_min = aug_size, aug_max = aug_size, aug_p = 1)
        
    elif transform_name == 'nlp_random_similar_word_insertion_word2vec_embedding':
        aug = naw.WordEmbsAug(model_type='word2vec', 
                              model_path=model_dir+'GoogleNews-vectors-negative300.bin',
                              action="insert")
        
        
    elif transform_name == 'nlp_random_similar_word_insertion_glove_embedding':
        aug = naw.WordEmbsAug(model_type='glove', 
                              model_path=model_dir+'GoogleNews-vectors-negative300.bin',
                              action="insert")
        
    elif transform_name == 'nlp_random_similar_word_insertion_fasttext_embedding':
        aug = naw.WordEmbsAug(model_type='fasttext', 
                              model_path=model_dir+'GoogleNews-vectors-negative300.bin',
                              action="insert")
    
    elif transform_name == 'nlp_random_similar_word_substitution_word2vec_embedding':
        aug = naw.WordEmbsAug(model_type='word2vec', 
                              model_path=model_dir+'GoogleNews-vectors-negative300.bin',
                              action="substitute")
        
        
    elif transform_name == 'nlp_random_similar_word_substitution_glove_embedding':
        aug = naw.WordEmbsAug(model_type='glove', 
                              model_path=model_dir+'GoogleNews-vectors-negative300.bin',
                              action="substitute")
        
    elif transform_name == 'nlp_random_similar_word_substitution_fasttext_embedding':
        aug = naw.WordEmbsAug(model_type='fasttext', 
                              model_path=model_dir+'GoogleNews-vectors-negative300.bin',
                              action="substitute")
        
        
    elif transform_name == 'nlp_random_similar_word_substitution_tfidf_embedding':
        aug = naw.TfIdfAug(model_path=os.environ.get("MODEL_DIR"),
                           action="substitute")
        
        
    elif transform_name == 'nlp_random_contextual_word_insertion_bert_uncased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', 
                                        action="insert")
        
    elif transform_name == 'nlp_random_contextual_word_insertion_bert_cased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', 
                                        action="insert")
    
    elif transform_name == 'nlp_random_contextual_word_insertion_distilbert_uncased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', 
                                        action="insert")
        
    elif transform_name == 'nlp_random_contextual_word_insertion_distilbert_cased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', 
                                        action="insert")
        
    elif transform_name == 'nlp_random_contextual_word_insertion_roberta_base_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='roberta-base', 
                                        action="insert")
        
    elif transform_name == 'nlp_random_contextual_word_insertion_distilroberta_base_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='distilroberta-base', 
                                        action="insert")
    
    elif transform_name == 'nlp_random_contextual_word_insertion_xlnet_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='xlnet', 
                                        action="insert")
    
    elif transform_name == 'nlp_random_contextual_word_insertion_bart_base_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='facebook/bart-base', 
                                        action="insert")
        
    elif transform_name == 'nlp_random_contextual_word_insertion_squeezebert_uncased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='squeezebert/squeezebert-uncased', 
                                        action="insert")
    
    
    elif transform_name == 'nlp_random_contextual_word_substitution_bert_uncased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', 
                                        action="substitute")
   
    elif transform_name == 'nlp_random_contextual_word_substitution_bert_cased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='bert-base-cased', 
                                        action="substitute")
        
    elif transform_name == 'nlp_random_contextual_word_substitution_distilbert_uncased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', 
                                        action="substitute")
   
    elif transform_name == 'nlp_random_contextual_word_substitution_distilbert_cased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-cased', 
                                        action="substitute")
        
    elif transform_name == 'nlp_random_contextual_word_substitution_roberta_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='roberta-base', 
                                        action="substitute")
  
    elif transform_name == 'nlp_random_contextual_word_substitution_distilroberta_base_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='distilroberta-base', 
                                        action="substitute")
        
    elif transform_name == 'nlp_random_contextual_word_substitution_xlnet_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='xlnet', 
                                        action="substitute")

    elif transform_name == 'nlp_random_contextual_word_substitution_bart_base_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='facebook/bart-base', 
                                        action="substitute")
        
    elif transform_name == 'nlp_random_contextual_word_substitution_squeezebert_uncased_embedding':
        aug = naw.ContextualWordEmbsAug(model_path='squeezebert/squeezebert-uncased', 
                                        action="substitute")
        
        
    elif transform_name == 'nlp_wordnet_synonym':
        aug = naw.SynonymAug(aug_min = aug_size
                             , aug_max = aug_size 
                             , aug_p = 1)
    
    elif transform_name == 'nlp_ppdb_synonym':
        aug = naw.SynonymAug(aug_src='ppdb'
                             , model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-s-all'
                             , aug_min = aug_size, aug_max = aug_size, aug_p = 1)
        
    elif transform_name == 'nlp_antonym':
        aug = naw.AntonymAug(aug_min = aug_size
                             , aug_max = aug_size 
                             , aug_p = 1)
        
    elif transform_name == 'nlp_random_word_swap':
        aug = naw.RandomWordAug(action = 'swap'
                               , aug_min = aug_size
                               , aug_max = aug_size 
                               , aug_p = 1)
        
    elif transform_name == 'nlp_random_word_delete':
        aug = naw.RandomWordAug(aug_min = aug_size
                                , aug_max = aug_size 
                                , aug_p = 1)
    
    elif transform_name == 'nlp_random_crop':
        aug = naw.RandomWordAug(action = 'crop'
                               , aug_min = aug_size
                               , aug_max = aug_size 
                               , aug_p = 1)
        
    elif transform_name == 'nlp_random_token_split':
        aug = naw.SplitAug(aug_min = aug_size
                           , aug_max = aug_size 
                           , aug_p = 1)
        
        
    # HOW TO ADD USER INPUT HERE?
    # All possible models: https://huggingface.co/models?filter=translation&search=Helsinki-NLP
    elif transform_name == 'nlp_back_translation_aug':
        aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', 
                                     to_model_name='facebook/wmt19-de-en')
    
    # Reserved Word Augmenter??
    
    elif transform_name == 'nlp_contextual_sentence_insertion_gpt2_embedding':
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')
   
    elif transform_name == 'nlp_contextual_sentence_insertion_xlnet_cased_embedding':
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='xlnet-base-cased')
    
    elif transform_name == 'nlp_contextual_sentence_insertion_distilgpt2_embedding':
        aug = nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
 
    elif transform_name == 'nlp_abstractive_summarization_bart_large_cnn':
        aug = nas.AbstSummAug(model_path='facebook/bart-large-cnn')
        
    elif transform_name == 'nlp_abstractive_summarization_t5_small':
        aug = nas.AbstSummAug(model_path='t5-small')
 
    elif transform_name == 'nlp_abstractive_summarization_t5_base':
        aug = nas.AbstSummAug(model_path='t5-base')

    elif transform_name == 'nlp_abstractive_summarization_t5_large':
        aug = nas.AbstSummAug(model_path='t5-large')
        
    return aug.augment

    
    
def initialize_bert_transform(config, aug_size, num_aug_samples = 1):
    assert 'bert' in config.model
    assert config.max_token_length is not None

    tokenizer = getBertTokenizer(config.model)
    print("bert:", aug_size)
    def transform(text):
        
        if 'nlp' in config.transform:
            transform_name = config.transform
            #print(text)
            
            aug = initialize_nlpaug_transform(transform_name, aug_size)
            
            
            text = aug(text, n = num_aug_samples)
            #print(text)
            #print(aug_size)
            
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=config.max_token_length,
            return_tensors='pt')
        
        if config.model == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif config.model == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer

def initialize_image_base_transform(config, dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(dataset.original_resolution)!=max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    if config.target_resolution is not None and config.dataset!='fmow':
        transform_steps.append(transforms.Resize(config.target_resolution))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform = transforms.Compose(transform_steps)
    return transform

def initialize_image_resize_and_center_crop_transform(config, dataset):
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(int(res*config.resize_scale) for res in dataset.original_resolution)
    if config.target_resolution is not None:
        target_resolution = config.target_resolution
    else:
        target_resolution = dataset.original_resolution
    transform = transforms.Compose([
        transforms.Resize(scaled_resolution),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def initialize_poverty_transform(is_training):
    if is_training:
        transforms_ls = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
            transforms.ToTensor()]
        rgb_transform = transforms.Compose(transforms_ls)

        def transform_rgb(img):
            # bgr to rgb and back to bgr
            img[:3] = rgb_transform(img[:3][[2,1,0]])[[2,1,0]]
            return img
        transform = transforms.Lambda(lambda x: transform_rgb(x))
        return transform
    else:
        return None

def initialize_rxrx1_transform(is_training):
    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform