# unicode, regex, json for text digestion
import unicodedata
import re
import json
import os

# nltk: natural language toolkit -> tokenization, stopwords
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

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
    string = re.sub(r"[^a-z0-9'\s]", ' ', string)
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
    additional_stopwords = ['github', 'http', 'code']
    nltk.download('wordnet')
    nltk.download('stopwords')
    stopword_list = stopwords.words('english') + additional_stopwords
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

def create_final_csv():
    filename = 'final_data.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)

    else:
        file = open('data.json')

        data = json.load(file)
        data = pd.DataFrame(data)


        data = data.assign(cleaned=data.readme_contents.apply(basic_clean))
        data = data.assign(without_stop_words=data.cleaned.apply(remove_stopwords))
        data = data.assign(tokenized=data.without_stop_words.apply(tokenize))\
                .assign(cleaned= data.without_stop_words.apply(remove_stopwords))\
                .assign(stem=data.without_stop_words.apply(stem))\
                .assign(lemm=data.without_stop_words.apply(lemmatize))

        excluded_languages = list(data.language.value_counts()[data.language.value_counts() < 6].index)
        data =data[~data.language.isin(excluded_languages)]

        data.to_csv(filename, index=False)

        return data

if __name__ == '__main__':
    data = create_final_csv()
    print(data)
    print(data.size)

