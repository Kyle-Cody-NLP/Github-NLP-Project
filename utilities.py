import pandas as pd
import modeling as md
import split
import numpy as np
import nltk
from itertools import chain,cycle
from IPython.display import display_html
import utilities
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


def most_frequent_words(df):

    lang_dict = {}
    languages = df.language.unique()
    for lang in languages:
        lang_dict[lang] = df[df.language==lang]


    one_string_per_language = {}
    for lang in languages:
        one_string_per_language[lang] = ""
        for val in list(lang_dict[lang].lemm.values):
            one_string_per_language[lang] += val


    for key in one_string_per_language.keys():
        one_string_per_language[key] = one_string_per_language[key].split(' ')


    list_of_word_counts = []
    all_strings = ''

    for key in one_string_per_language.keys():
        list_of_word_counts.append(pd.Series(one_string_per_language[key], name=key).value_counts())
        for val in one_string_per_language[key]:
            all_strings += f' {val}'

    all_strings = pd.Series(all_strings.split(' '), name='all_words').value_counts()
    list_of_word_counts.append(all_strings)
    languages = np.append(languages, 'all_words')

    freq_df = (pd.concat(list_of_word_counts, axis=1, sort=True)
                .set_axis(languages, axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    
    stop_words = ['0', '1', '2', 'http', 'com', 'org']
    
    df2 = freq_df.drop(index=stop_words)
    top_words = df2.sort_values(by='all_words', ascending=False).head(11).all_words
    top_words = pd.DataFrame(top_words)
    
    return top_words


def make_top_bigrams(df):
    
    bigrams = list(nltk.ngrams(" ".join(df.lemm.astype(str)).split(), 2))
    bigrams = pd.Series(bigrams).value_counts().head(50)

    top_bigrams = []
    
    for ind in list(bigrams.head(20).index):
        if 'http' not in " ".join(ind):
            top_bigrams.append(ind)
            
    return pd.DataFrame({'top_bigrams': top_bigrams})


def make_top_trigrams(df):
    
    trigrams = list(nltk.ngrams(" ".join(df.lemm.astype(str)).split(), 3))
    trigrams = pd.Series(trigrams).value_counts().head(50)

    top_trigrams = []
    
    for ind in list(trigrams.head(20).index):
        if 'http' not in " ".join(ind):
            top_trigrams.append(ind)
            
    return pd.DataFrame({'top_trigrams': top_trigrams})


def display_side_by_side(*args,titles=cycle([''])):
    '''
    This allows the display of two or more DataFrame tables side by side.
    '''
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)

def plot_top_words(top_words):
    top_words.plot(kind='bar')
    plt.title('Top 10 Occurring Words', color='black')
    plt.tick_params(axis='x', labelrotation=45)
    plt.tick_params(axis='y')
    plt.subplots_adjust(bottom=0.25)

    plt.savefig('top_10_words.png', transparent=True)


def make_language_dict(df):
    lang_dict = {}
    for lang in df.language.unique():
        lang_dict[lang] = df.lemm[df.language == lang]
        lang_dict[lang] = pd.Series((" ".join(lang_dict[lang])).split(), name=lang).value_counts().head(15)
        
    return lang_dict


def loop_n_times_knn(train, validate, test, top=15):
    train_accuracies = {}
    validate_accuracies = {}
    
    for i in range(1,top):
        knn = KNeighborsClassifier(n_neighbors=i)

        x_train, y_train, x_validate, y_validate, x_test, y_test = md.xy_train_validate_test(train, validate, test, 'programming_language_99')
        knn.fit(x_train, y_train)

        train_actual = y_train
        validate_actual = y_validate
        test_actual = y_test

        train_prediction = knn.predict(x_train)
        validate_prediction = knn.predict(x_validate)
        test_prediction = knn.predict(x_test)
        
        train_accuracies[i] = accuracy_score(train_actual, train_prediction)
        validate_accuracies[i] = accuracy_score(validate_actual, validate_prediction)

    return train_accuracies, validate_accuracies

def create_encoded_df(df):
    tfidf = TfidfVectorizer(ngram_range=(1,3))
    tfidfs = tfidf.fit_transform(df.dropna().lemm.values)

    tfidf_df = pd.DataFrame(tfidfs.todense(), columns=tfidf.get_feature_names())
    col = pd.DataFrame({'programming_language_99': df.dropna().reset_index().drop(columns='index').language.values})

    encoded_df = pd.concat([tfidf_df, col], axis=1)

    return encoded_df

def train_val_test_knn(train,validate, test):

    knn = KNeighborsClassifier(n_neighbors=11)

    x_train, y_train, x_validate, y_validate, x_test, y_test = md.xy_train_validate_test(train, validate, test, 'programming_language_99')
    knn.fit(x_train, y_train)

    train_actual = y_train
    validate_actual = y_validate
    test_actual = y_test

    train_prediction = knn.predict(x_train)
    validate_prediction = knn.predict(x_validate)
    test_prediction = knn.predict(x_test)

    print(f'Train Accuracy: {round(accuracy_score(train_actual, train_prediction) * 100, 2)}%')
    print(f'Validate Accuracy: {round(accuracy_score(validate_actual, validate_prediction), 4) * 100}%')


def loop_n_times_knn(train, validate, test, top=15):
    train_accuracies = {}
    validate_accuracies = {}
    
    for i in range(1,top):
        knn = KNeighborsClassifier(n_neighbors=i)

        x_train, y_train, x_validate, y_validate, x_test, y_test = md.xy_train_validate_test(train, validate, test, 'programming_language_99')
        knn.fit(x_train, y_train)

        train_actual = y_train
        validate_actual = y_validate
        test_actual = y_test

        train_prediction = knn.predict(x_train)
        validate_prediction = knn.predict(x_validate)
        test_prediction = knn.predict(x_test)
        
        train_accuracies[i] = accuracy_score(train_actual, train_prediction)
        validate_accuracies[i] = accuracy_score(validate_actual, validate_prediction)

    return train_accuracies, validate_accuracies


def create_results(train, validate, test):
    train_acc, validate_acc = loop_n_times_knn(train, validate, test)

    train_acc_df= pd.DataFrame(data={'knn': train_acc.keys(), 'train_accuracy': train_acc.values()})
    validate_acc_df = pd.DataFrame(data={'validate_accuracy': validate_acc.values()})
    results = pd.concat([train_acc_df, validate_acc_df], axis=1)
    
    return results

def plot_results(results):
    plt.plot(results.knn, results.train_accuracy, color='black')
    plt.plot(results.knn, results.validate_accuracy, color='green')
    plt.xlabel('n Nearest Neighbors', color='green')
    plt.ylabel('% Accuracy', color='green')
    plt.xticks(color='green')
    plt.yticks(color='green')
    plt.legend(['train', 'validate'])
    plt.vlines(x=11, ymin=.2, ymax=1, linestyles='dotted', color='blue')
    plt.title('Train Vs Validate Performance', color='green')
    plt.savefig('trainVsvalidate.png', transparent=True)
    