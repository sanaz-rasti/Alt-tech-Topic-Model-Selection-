''' --------------------------------------------------------------------- '''
''' -----------------------  Import Dependencies ------------------------ '''

from sklearn.feature_extraction import text
from ctypes.wintypes import WORD

import pandas as pd
import re
import logging as ping
from sqlalchemy import create_engine

import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

''' --------------------------------------------------------------------- '''
''' ---------------------------  Channel Data --------------------------- '''
''' There are two separate Tables for each channel of data in our dataset, 
    These tables are read and appended in one as the orig text corpus. 
    The object take list of target channel IDs. 

    ChannelData class takes 3-args of:
        - datapath: string, pointing at '.db' file location 
        - list_6irish_channels,
        - target_table_1: str('Target Table_1'),
        - target_table_2: str('Target Table_2')):

    Calling corpus() method returns:
        - TextCorpus = the original message content '''

class ChannellData:

    TextCorpus = pd.DataFrame()

    def __init__(self,
                 datapath: str('Path to database directory'),
                 list_6irish_channels,
                 target_table_1: str('Target Table_1'),
                 target_table_2: str('Target Table_2')):
        
        self.datapath = datapath
        self.list_6irish_channels = list_6irish_channels
        self.target_table_1 = target_table_1
        self.target_table_2 = target_table_2
        
    def corpus(self):

        ping.info('Saving recipes to {}'.format(self.datapath))
        db = create_engine('sqlite:///{}'.format(self.datapath))
        
        
        text_content_1 = pd.read_sql(self.target_table_1,con=db)
        text_content_1 = text_content_1.loc[text_content_1["Channel_ID"].isin(self.list_6irish_channels)]
        
        text_content_2 = pd.read_sql(self.target_table_2,con=db)
        text_content_2 = text_content_2.loc[text_content_2["Channel_ID"].isin(self.list_6irish_channels)]
        
        self.__class__.TextCorpus = text_content_1.append(text_content_2)
        return self.__class__.TextCorpus


''' ---------------------------------------------------------------------- '''
''' ---------------------------  Data Cleaning  -------------------------- '''
''' 
    


 '''



class DataCleaning:
    
    datapath = r'newdb.db'
    db = create_engine('sqlite:///{}'.format(datapath))
    TextCorpus_unique = pd.DataFrame()
    lemstem_text_Content = ''
    len_textcorpus = 0

    def __init__(self, 
                 orig_corpus: str('Original Corpus in form of DataFrame'), 
                 output_table: str('Name of output table')) -> None:
        self.orig_corpus = orig_corpus
        self.output_table = output_table

    
    #----------------------------------------
    @staticmethod
    def get_wordnet_pos(word):

        # Mapping Part of Speech tag to the first character that lemmatize() accepts
        tag = nltk.pos_tag([word])[0][1][0].upper()
        
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV,
                    "S": wordnet.ADJ_SAT}
        
        if tag in tag_dict.keys():
            return tag_dict.get(tag, tag_dict[tag])
    
        else:
            return tag_dict.get(tag, wordnet.NOUN)
    
    
    #----------------------------------------
    def StemLemmatizerText(self, text_Content):
        h=list()
        h[:]=[]

        text_Content= text_Content.lower()

        text_Content = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text_Content)

        text_Content = re.sub('https?[:;]?/?/?\S*', ' ', text_Content)

        text_Content = re.sub(r'#|!|@|,|•|“|”|\"|\'|’s|n’t|’','',text_Content)

        text_Content = re.sub(r'\?',' ',text_Content)

        text_Content = re.sub('[.] *',' ',text_Content)

        text_Content = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|[0-9]?[0-9]?(\.[0-9][0-9]?)?|(\w+:\/\/\S+)|^rt", "", text_Content)

        text_Content = re.sub(r'com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[ ]?',' ',text_Content)


        for token in [WordNetLemmatizer().lemmatize(w, pos = self.__class__.get_wordnet_pos(w)) for w in nltk.word_tokenize(text_Content)]:
            if wordnet.morphy(token):
                token=wordnet.morphy(token)

            if self.__class__.get_wordnet_pos(token)==wordnet.NOUN:
                 token =  PorterStemmer().stem(token)

            if not token.isdigit() and token not in text.ENGLISH_STOP_WORDS:
                h.append(WordNetLemmatizer().lemmatize(token, pos = self.__class__.get_wordnet_pos(token)))

        stemed_text_Content=''            
        stemed_text_Content=' '.join(h)
        return stemed_text_Content
    
    
    #----------------------------------------
    def refine_specific_char(self):
        TextCorpus_unique = self.orig_corpus.drop_duplicates(subset = ['Channel_ID', 'MessageContent', 'MessageID', 'MessageDate', 'Message_FROm_UserID'], keep = 'first')
        Text_Corpus_Refine = TextCorpus_unique
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.match(r'https?[:;]?/?/?\S*')) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("Tnx for joining")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("Thanks for joining")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("Thank you to all the new members for joining")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("Thank u x")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("Thanks for joining our small group")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("Thanks for Joining")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains(" has been banned! Reason: CAS ban")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("SEND OUR QR CODE")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("SHARE OUR QR CODE")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.contains("send our QR code")) ]
        Text_Corpus_Refine = Text_Corpus_Refine.loc[ ~(Text_Corpus_Refine['MessageContent'].str.match("Check this out.."))]
        self.__class__.TextCorpus_unique = Text_Corpus_Refine.reset_index()
        self.__class__.len_textcorpus = len(Text_Corpus_Refine)


    #----------------------------------------
    def lemma_stem(self):
        step = 10000
        strt = 0
        end = strt + step
        df = self.__class__.TextCorpus_unique
        
        while end < self.__class__.len_textcorpus:
    
            temp_df = df.loc[strt:end]
            temp_df["strip_text"] = ''
            temp_df["semantic_unit_count"] = ''

            for i in range(strt, end+1):
                if temp_df.at[i,"MessageContent"] and self.__class__.StemLemmatizerText(self, temp_df.at[i,"MessageContent"]):
                    temp_df.at[i,"strip_text"] = self.__class__.StemLemmatizerText(self,temp_df.at[i,"MessageContent"])
                    temp_df.at[i,"semantic_unit_count"] = len(nltk.word_tokenize(temp_df.at[i,"strip_text"]))
    
            temp_df.to_sql(self.output_table,
                           con=self.__class__.db, 
                           index=False, 
                           if_exists="replace")
    
            strt = end + 1
            end  = end + step    
            del(temp_df)

''' ------------------------------------ '''    
''' --------------- Main --------------- '''
def main():
    lst_ich = [-1001250192398,
               -1001486710610,
               -1001719423078, 
               -1001522159637,
               -1001242115487,
               -1001198445854]
    
    chd = ChannellData(datapath = r'TelegramData.db',
                        list_6irish_channels = lst_ich,
                        target_table_1 = "TextContent_Telegram_Channel",
                        target_table_2 = "TextContent_Telegram_Channel_2")

    orig_msg_content = chd.corpus()

    # create data cleaning object
    dco = DataCleaning(orig_msg_content, output_table = "Strip_Text")
    
    dco.lemma_stem()
    

''' ---------------------------------------'''
if __name__ == "__main__":
    main()