#importing necessary modules
import pandas as pd
import contractions 
import re
from tqdm import tqdm
import sys
import nltk
import unicodedata
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sacremoses import MosesDetokenizer

#expands english contractions(i'd-->i would)
#renmoves text between '(some text here)' and '[some text here]'
def remove_noise(in_data):

    ''' Denoises Data:\n
    Expands english word contractions.\n
    Removes text between brackets () and [].\n
    Returns denoised data list.''' 

    denoised_data  = []

    for row in in_data:
        
        contracted_row = contractions.fix(row)

        filtred_text = re.sub("[\(\[].*?[\)\]]", "", contracted_row)

        denoised_data.append(filtred_text)
       
        
    return denoised_data

#converts text data to token
def tokenize_corpus(text_corpus):

    '''Tokenize Data:\n
        Converts text data into tokens.\n
        Returns tokens data in a list.
        '''

    tokenized_data = []

    for row in text_corpus:

        token = nltk.word_tokenize(row)

        tokenized_data.append(token)
    
    return tokenized_data

#removes non-ASCII characters from the tokenized data
def remove_foreign_characters(text_tokens):

    '''Remove non-ASCII charcters:\n
       Removes foreign non ASCII charcters\n
       inspecting the tokens.\n
       Returns unicode string tokens in a list.''' 

    filtered_tokens = []

    for token in text_tokens:

        filtered_token = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        filtered_tokens.append(filtered_token)

    return filtered_tokens

#comverts all characteres to lowercase 
def set_lowercase(text_tokens):

    '''To lowercase:\n
      Converts all string tokens to lowercase.\n
      Returns lowercase string tokens in a list.'''    


    filtered_tokens = []

    for token in text_tokens:

        filtered_token = token.lower()

        filtered_tokens.append(filtered_token)

    return filtered_tokens

#removes punctuation
def remove_punctuation(text_tokens):

    '''Removes Punctuation:\n
       Removes punctuation from the token data.\n
       Returns punctuation free list of string tokens.'''    

    filtered_tokens = []
    
    for token in text_tokens:

        filtered_token = re.sub(r'[^\w\s]', '', token)
        
        if filtered_token != '':
            filtered_tokens.append(filtered_token)
        
    return filtered_tokens

#replace numerical values to its corresponding numbers in word
def replace_number_to_words(text_tokens):

    '''Replaces number:\n
       Replaces digit with its corresponding number in word.\n
       Returns list of tokens.''' 


    #getting inflection engine for digit to word conversion
    inf_engine = inflect.engine()

    filtered_tokens = []

    for token in text_tokens:

        if token.isdigit():

            filtered_token = inf_engine.number_to_words(token)

            filtered_tokens.append(filtered_token)

        else:

            filtered_tokens.append(token)

    return filtered_tokens

def stem_tokens(text_tokens):

    '''Token Stemming:\n
        Performs Stemming in string tokens.\n
        Returns stemmed string tokens in a list.'''    


    stemmed_tokens = []

    stemmer = LancasterStemmer()

    for token in text_tokens:

        stemmed = stemmer.stem(token)

        stemmed_tokens.append(stemmed)

    return stemmed_tokens

def lemmatize_tokens(text_tokens):
    '''Token lemmatization:\n
       Performs lemmatization in string tokens (only verb).\n
       Retrurns lemmatized string tokens.'''    


    lemmatized_tokens = []

    lemmatizer  =  WordNetLemmatizer()

    for token in text_tokens:

        lemmatized = lemmatizer.lemmatize(token,pos='v')

        lemmatized_tokens.append(lemmatized)
    
    return lemmatized_tokens


#normalize data from the total token corpus
def normalize_data(token_corpus):

    '''Normalize tokens:\n
    1. Removes Foreign characters.
    2. Converts tokens to lowercase.
    3. Removes punctuation.
    4. Performs stemming.
    5. Performs lemmatization.\n

    Returns list of list of tokens.'''
    
    normalized_token_data = []

    for one_list in tqdm(token_corpus):
        
        tokens = remove_foreign_characters(one_list)
        tokens = set_lowercase(tokens)
        tokens = remove_punctuation(tokens)
        tokens = replace_number_to_words(tokens)
        tokens = lemmatize_tokens(tokens)
        tokens = stem_tokens(tokens)

        normalized_token_data.append(tokens)
    
    return normalized_token_data

#performs all steps of data preprocessing
def perform_pre_processing(data):
    ''' Data preprocessing:\n
        1. Removes data noise.
        2. Tokenizes data.
        3. Normalize the token data.\n
         Returns list of list of tokens.   
        '''
    
    denoised_data = remove_noise(data)
    tokenized_data = tokenize_corpus(denoised_data)
    normalized_data = normalize_data(tokenized_data)
    
    return tokenized_data,normalized_data


def detokenize_data(token_list):

    detokenized_sentences = []

    detokenizer = MosesDetokenizer()

    for one_document_token in token_list:

        sentence = detokenizer.detokenize(one_document_token, return_str=True)

        detokenized_sentences.append(sentence)

    return detokenized_sentences

def tag_tokens_by_pos(tokens):
    pos_tagged_tokens = []
    pos_tagged_tokens_list = []
    tagged_tokens = [nltk.pos_tag(token) for token in tokens]
    
    for tupletoken in tagged_tokens:
        for apair in tupletoken:
            token,tag = apair
            pos_tagged_tokens.append(token+'_'+tag )
        pos_tagged_tokens_list.append(pos_tagged_tokens)
    
    return pos_tagged_tokens_list
        




    





    



















