# unicode, regex, json for text digestion
import unicodedata
import re
import json

# nltk: natural language toolkit -> tokenization, stopwords
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# pandas dataframe manipulation, acquire script, time formatting
import pandas as pd
import acquire
from time import strftime

# shh, down in front
import warnings
warnings.filterwarnings('ignore')





def basic_clean(string):
    string = string.lower()
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    string = string.replace
    string = re.sub(r"[^a-z0-9'\s]", '', df)
    return string


def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = (tokenizer.tokenize(string, return_str=True))
    return string


def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    
    return string


def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
        lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)
    return string


def remove_stopwords(string, extra_words = [], exclude_words = []):
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords


def split_github_data(df):
    '''
    Takes in a cleaned github dataframe, splits it into train, validate and test subgroups and then returns those subgroups.
    Arguments: df - a cleaned pandas dataframe with the expected feature names and columns in the github dataset
    Return: train, validate, test - dataframes ready for the exploration and model phases.
    '''

    train_validate, test = train_test_split(df, test_size=.2, 
        random_state=17)

    train, validate = train_test_split(train_validate, test_size=.3, 
        random_state=17)
    return train, validate, test