'''--------------------------- Dependencies and defined Classes  ---------------------------'''
from IPython.core.display import display, HTML
import pandas as pd
import re
from termcolor import *
import spacy

nlp = spacy.load('en_core_web_sm')
import numpy as np
from functools import wraps
import re
from ctypes.wintypes import WORD

# Gensim
from gensim.parsing.preprocessing import *
from gensim.similarities import Similarity
from gensim.test.utils import get_tmpfile
import gensim.corpora as corpora
from sklearn.feature_extraction import text

# NLTK
from nltk.text import Text
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer     
nltk.download('punkt')
nltk.download('stopwords')


''' ----------------------------------------------------- ''' 
''' ------------------- Data Cleaning ------------------- '''
''' 
    INPUT for the Data Cleaning class:
        - The path to original corpus: corpus which is list of documnets and is stored as .txt file
        - The path to list of stopwords.
    
    OUTPUT:
        - after creating the object and calling the data_cleaning() method on the object, 
        two files are created as the outcome of this function:
            - clean_corp.txt -> is a .txt file containing list of cleaned documents
            - orig_corp.txt  -> is a .txt file containing list of original documnets
            
    The aim of producing the orig_corp file is to remove the documents which are empty in the        cleaned corpus. This leads to the oorig_corp and clean_corp to be the same size 
'''

class data_cleaining:
    
    def __init__(self, 
                 path_to_corpus: ".txt file containing list of documents.",
                 stopwords: "path to list of stopwords."):
        
        with open(path_to_corpus, "r") as f:
            corpus = f.readlines()
        
        self.corpus    = corpus
        self.stopwords = stopwords
        
        print(f'Length of original Corpus is: {len(corpus)}')
        
        
    def data_cleaning(self):
        
        StopWords = stopwords.words('english')
        StopWords.extend(list(np.loadtxt(self.stopwords, dtype='str')))
        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    u"\U00002500-\U00002BEF"  # chinese char
                                    u"\U00002702-\U000027B0"
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    u"\U0001f926-\U0001f937"
                                    u"\U00010000-\U0010ffff"
                                    u"\u2640-\u2642" 
                                    u"\u2600-\u2B55"
                                    u"\u200d"
                                    u"\u23cf"
                                    u"\u23e9"
                                    u"\u231a"
                                    u"\ufe0f"  # dingbats
                                    u"\u3030"
                                    "]+", flags=re.UNICODE)

        corpus_copy = self.corpus
        # looking for clean_corp
        clean_corp  = []
        
        for txt in self.corpus:
            text = txt
            txt  = str(txt)
            txt  = lower_to_unicode(txt)
            txt  = strip_numeric(txt)
            txt  = emoji_pattern.sub(r'', txt)
            txt  = strip_punctuation(txt)
            txt  = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags = re.MULTILINE)
            txt  = re.sub(r'http\S+', '', txt)
            txt  = remove_stopwords(txt, stopwords = StopWords)
            txt  = ''.join(txt)

            if txt == '':
                corpus_copy.remove(text)
            else:
                clean_corp.append(txt)

        clean_corp  = list(filter(None, clean_corp))
        corpus_copy = list(filter(None, corpus_copy))
        
        with open('clean_corp.txt', "w") as f:
            for l in clean_corp:
                f.write(l + '\n')
                
        with open('orig_corp.txt', "w") as f:
            for l in corpus_copy:
                f.write(l + '\n')
        
            
''' -------------------------------------- '''    
''' ---------------- Main ---------------- '''
def main():
    # example of path to corpus
    # path_to_corp = 'corpus.txt'
    path_to_corp = input('Enter the path to the corpus: ')

    # path to stopwords
    # stopwordpath = 'stopwords.txt'
    stopwordpath = input('Enter the path to list of stopwords: ')

    dc_obj = data_cleaining(path_to_corp, stopwordpath)

    dc_obj.data_cleaning()
    

''' ---------------------------------------'''
if __name__ == "__main__":  
    main()
    
    
    