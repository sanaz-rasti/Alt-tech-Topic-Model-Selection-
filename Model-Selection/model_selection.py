'''--------------------------- Dependencies ---------------------------'''
from gensim.models import LdaSeqModel
from gensim.test.utils import common_texts, common_dictionary
from gensim.corpora import Dictionary
'''--------------------------- Dependencies and defined Classes  ---------------------------'''
from IPython.core.display import display, HTML
import warnings
from datetime import datetime
import pandas as pd
import re
import datetime
import os
import numpy as np
import seaborn as sns
from functools import wraps
import time
import logging as ping
from sqlalchemy import create_engine
from IPython.display import display
from hdbscan import HDBSCAN

from typing import TypeVar, Generic, Tuple
import statistics
import matplotlib.pyplot as plt
from ctypes.wintypes import WORD
from gensim.corpora import Dictionary
import pprint

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import rand_score

# Gensim 
from gensim.similarities import Similarity
from gensim.test.utils import get_tmpfile
import gensim.corpora as corpora
from sklearn.feature_extraction import text
from gensim.parsing.preprocessing import *

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

# Topic model
from bertopic import BERTopic

# Dimension reduction
from umap import UMAP

display(HTML("<style>.container { width:100% !important; }</style>"))
pd.options.display.max_colwidth = 200
warnings.filterwarnings('ignore')

''' --------------------------- Doc Topic Similarity Class --------------------------- '''
'''
DESCRIPTION: 
    DocTopicSim class includes 3-major methods with following specs: 
        - lda_doc_topic_sim: produces models with different Ngram-ranges and choose best performing model using Document-Topic-Similarity
        - nmf_doc_topic_sim: produces models with different Ngram-ranges and choose best performing model using Document-Topic-Similarity
        - ntm_doc_topic_sim: produces two neural topic model, employing Contextualized Topic Modeling.
        - Description of model selection procedure can be found in our paper titled:
            "ALT-TECH TOPIC MODELING INCORPORATING A TOPIC MODEL SELECTION STRATEGY"

        Each method returns a descriptive DataFrame of topic models
        The class also includes 2-minor methods:
        - save_Top_Doc: For each topic model, saves Topics and Documents for further investigation
        - plot_DTS_RS: This method should be called only if the dataset is labelled. 
            After producing all three topic models, the DTS-RS of all models are plotted  

INPUT:
    The input for the class requires:
                    - orig_msg_content
                    - strip_text
                    - max_ngram_range
                    - StopWords text file
                    - Number of topics: ntopics
                    - Number of terms per topic: nterms
                    - Number of relevant documents to be assigned from each extracted topic: ndocs
                                                     
                                                                            
        
EXAMPLE:
    - Read origCorp and cleanedCorp from your dataset
    - Produce the dts object 

    dts = DocTopicSim(chnlN = 'ChannelName',
                      orig_msg_content = origCorp,
                      strip_text = cleanedCorp,
                      max_ngram_range = 5,
                      targets = [],
                      ntopics = 10,
                      nterms = 10,
                      ndocs = 5)
    
    dts.lda_doc_topic_sim()
    df = dts.df_lda
    df.to_csv('chName10topic_lda.csv', index=False)

    dts.nmf_doc_topic_sim()
    df = dts.df_nmf
    df.to_csv('chName10topic_nmf.csv', index=False)

    dts.ntm_doc_topic_sim()
    df = dts.df_ntm
    df.to_csv('chName10topic_ntm.csv', index=False)

    if len(targets) > 0 :
        dts.plot_DTS_RS()
   
'''
''' ---------------------------  Doc Topic Similarity --------------------------- '''
class DocTopicSim:
    
    doc_terms_lda  = []
    doc_terms_nmf  = []
    
    df_lda  = pd.DataFrame()
    df_nmf  = pd.DataFrame()
    df_ntm  = pd.DataFrame()
    
    dfp_lda  = pd.DataFrame()
    dfp_nmf  = pd.DataFrame()
    dfp_ntm  = pd.DataFrame()
    rs_ntm   = []
    avg_sims_ntm = []
    
    all_doc_targets_lda = pd.DataFrame()
    all_doc_targets_nmf = pd.DataFrame()
    all_doc_targets_ntm = pd.DataFrame()
    

    def __init__(self,
                 chnlN,
                 orig_msg_content: str('Should take the original meseg content from the database'),
                 strip_text: str('stripid text data from database'),
                 targets,
                 max_ngram_range,
                 ntopics: str('Number of desired topics to be extracted'),
                 nterms: str('Number of terms per topic'),
                 ndocs: str('Number of documents to be extracted')) -> None:
        
        self.chnlN = chnlN
        self.orig_msg_content = orig_msg_content
        self.strip_text = strip_text
        self.targets = targets
        self.max_ngram_range = max_ngram_range
        self.ntopics = ntopics
        self.nterms = nterms
        self.ndocs = ndocs
        self.NAllDocs = len(strip_text)
        
        
    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    '''--------- LDA Model: Similarity between "TopNDocs" to "Topic" -------------------'''
    def lda_doc_topic_sim(self):
        num_models = 0
        self.__class__.df_lda = pd.DataFrame()
        t = time.process_time()
        data = []
        dt1 = []
        
        for i in range(1,self.max_ngram_range):
            
            for j in range(1,self.max_ngram_range):
                
                if j >= i:
                    ngram = (i, j) 
                    vectorizer = CountVectorizer(max_df = 0.9, 
                                                min_df = 1, 
                                                max_features = 2000, 
                                                ngram_range = ngram)
                    
                    Atf = vectorizer.fit_transform(self.strip_text)
                    self.__class__.doc_terms_lda = vectorizer.get_feature_names()
                    
                    # topic model
                    lda = LatentDirichletAllocation(n_components = self.ntopics, random_state=1).fit(Atf)
                    
                    perplex = lda.bound_
                    
                    topic_results = lda.transform(Atf)

                    top_docs_indx = topic_results.argsort(axis=0)[-self.ndocs-1:]
                    
                    try:
                        del topics2present
                    except NameError:
                        pass
                    
                    topics2present = []
                    
                    for _, topic in enumerate(lda.components_):
                        top_terms = [self.__class__.doc_terms_lda[i]
                                     for i in topic.argsort()[-1:-self.nterms-1:-1]]
                        
                        topics2present.append(top_terms)
                        
                    print(f'Ngram_{ngram}.')
                    for k in range(self.ntopics):
                        
                        if len(self.targets) > 0 :
                            
                            tdo = [self.targets[x] for x in topic_results.argsort(axis=0)[::-1][:,k]]
                            
                            dr = [f'Model_{num_models}_{ngram}', k , tdo]
                            
                            dt1.append(dr)
                            
                        queryTopic = TreebankWordDetokenizer().detokenize(topics2present[k])
                        
                        for l in range(self.ndocs):

                            query_doc = self.strip_text[top_docs_indx[:, k][l]]
                            
                            doc1 = nlp(queryTopic)
                            
                            doc2 = nlp(query_doc)
                            
                            similarities = doc1.similarity(doc2)
                            print(f'Topic_{k} & Doc_{top_docs_indx[:, k][l]} : {similarities}.')
                            
                            if len(self.targets) > 0:
                                
                                data_row = [f'Model_{num_models}_{ngram}',
                                            k,
                                            f'Topic_{k}: {topics2present[k]}',
                                            f'Doc_{top_docs_indx[:,k][l]}: {self.orig_msg_content[top_docs_indx[:,k][l]]}',
                                            f'Doc_{top_docs_indx[:,k][l]}: {self.strip_text[top_docs_indx[:,k][l]]}',
                                            self.targets[top_docs_indx[:,k][l]],
                                            similarities,
                                           perplex]
                            else:
                                
                                data_row = [f'Model_{num_models}_{ngram}',
                                            k,
                                            f'Topic_{k}: {topics2present[k]}',
                                            f'Doc_{top_docs_indx[:,k][l]}: {self.orig_msg_content[top_docs_indx[:,k][l]]}',
                                            f'Doc_{top_docs_indx[:,k][l]}: {self.strip_text[top_docs_indx[:,k][l]]}',
                                            similarities]                   
                            
                            data.append(data_row)
                  
                    num_models += 1
                    
        if len(self.targets) > 0:
                        
            self.__class__.df_lda = pd.DataFrame(data, columns=['Model',
                                                                'Topic_num',
                                                                'Extracted_Topics',
                                                                'Relevant_Docs',
                                                                'Strip_text_of_doc',
                                                                'Ground_Truth_Target',
                                                                'Doc_Topic_Similarity',
                                                                'Perplexity'])
                        
                        
            self.__class__.all_doc_targets_lda = pd.DataFrame(dt1, columns = ['Model',
                                                                            'Topic_num',
                                                                            'all_doc_targets'])
                                     
        else:
            
            self.__class__.df_lda = pd.DataFrame(data, columns = ['Model',
                                                                'Topic_num',
                                                                'Extracted_Topics',
                                                                'Relevant_Docs',
                                                                'Strip_text_of_doc',
                                                                'Doc_Topic_Similarity'])
            
        
        # ------- Plot the model average similarities ------- #
        df = self.__class__.df_lda
        df.Doc_Topic_Similarity = df.Doc_Topic_Similarity.astype('float')
        numtopics = min(len(df['Topic_num'].drop_duplicates()), self.ntopics)
        lst = []
        index = []
        for m in range(num_models):
            
            model_name = df.Model[m*numtopics*self.ndocs]
            
            modelm_sim = [df.Doc_Topic_Similarity[i]
                          for i in range(len(df.Doc_Topic_Similarity))
                          if df.Model[i] == model_name]

            lst.append(statistics.mean(modelm_sim))
            index.append(model_name)
            print(f'Model Name: {model_name}: Average Sims: {statistics.mean(modelm_sim)}.')
            
            
        plt.bar(index, lst, width = 0.5)
        plt.xticks(rotation = 45)
        plt.margins(0.04)
        plt.ylabel("Similarity Index")
        plt.title("Doc Topic Similarity")
        
        # save and show
        plt.savefig(f'{self.chnlN}_LDA_DTS.png', dpi = 300, bbox_inches = 'tight')
        plt.show()
        
        # ------------
        df1  = pd.DataFrame(columns = ['Model_names', 'Average_Model_Sims'])
        df1['Model_names'] = index
        df1['Average_Model_Sims'] = lst
        lst = list(df1['Average_Model_Sims'])
        best_model  = df1['Model_names'][lst.index(max(lst))]
        worst_model = df1['Model_names'][lst.index(min(lst))]
        
        self.__class__.dfp_lda['AverageAllModelSims'] = lst
        self.__class__.dfp_lda['ModelNames'] = index
        
        if len(self.targets)>0:
            # ------------
            # For Best performing:
            # ------------
            df     = self.__class__.df_lda
            dfad   = self.__class__.all_doc_targets_lda
            target = list(set(self.targets))

            topic_nms = []

            numtopics = min(len(df['Topic_num'].drop_duplicates()),self.ntopics)

            for i in range(numtopics):
                topic_nms.append(f'T{i}')

            plotdata = {'Topics':topic_nms}

            for tar in target:
                plotdata[f'{tar}'] = [0]*numtopics 


            # --------- Rand Score, for all produced models
            modelname = list(df['Model'].drop_duplicates())
            rs = []
            labels_pred = []
            for k in range(numtopics):
                for j in range(self.ndocs):
                    labels_pred.append(f'T_{k}')

            for modelname in modelname:
                labels_true = list(df[df['Model'] == modelname].Ground_Truth_Target)
                rs.append(rand_score(labels_true, labels_pred))
                print(f'LDA: {modelname}. Rand Score:{rand_score(labels_true, labels_pred)}')
            self.__class__.dfp_lda['RandScoreAllModels'] = rs


            # ---------- Rand Score - plot best_performing -----------
            for k in range(numtopics):
                if len(self.targets) >0:
                    list_k = list(df['Ground_Truth_Target'].loc[(df['Model'] == best_model) & (df['Topic_num'] == k)])
                else:
                    list_k = list()

                for j in range(len(target)):

                    cnt = list_k.count(target[j])

                    plotdata[list(plotdata.keys())[j+1]][k] = cnt

            df_plot = pd.DataFrame(data = plotdata)
            df_plot.plot.bar(x = "Topics" , y = list(plotdata.keys())[1:], rot = 0, title = f'The Clustering Performance from \n LDA Best Performing Selected Model as {best_model}.')
            plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            plt.savefig(f'{self.chnlN}_LDAClusterBestModel_{best_model}.png')
            plt.show(block = True) 

            labels_pred = []
            for k in range(numtopics):
                for j in range(self.ndocs):
                    labels_pred.append(f'T_{k}')

            labels_true = list(df[df['Model'] == best_model].Ground_Truth_Target)

            print(f'Best performing model name: {best_model}')
            print(f'Labels True: {labels_true}')
            print(f'Labels Pred: {labels_pred}')
            print(f'Rand score: {(rand_score(labels_true, labels_pred))}')
            print(f'Rand score: {(adjusted_rand_score(labels_true, labels_pred))}')



            # ---------- Rand Score - plot worst_performing -----------
            for k in range(numtopics):

                list_k = list(df['Ground_Truth_Target'].loc[(df['Model'] == worst_model) & (df['Topic_num'] == k)])

                for j in range(len(target)):

                    cnt = list_k.count(target[j])

                    plotdata[list(plotdata.keys())[j+1]][k] = cnt

            df_plot = pd.DataFrame(data = plotdata)
            df_plot.plot.bar(x = "Topics" , y = list(plotdata.keys())[1:], rot = 0, title = f'The Clustering Performance from \n LDA Worst Performing Selected Model as {worst_model}.')
            plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            plt.savefig(f'{self.chnlN}_LDAClusterWorstModel_{worst_model}.png')
            plt.show(block = True) 

            labels_pred = []
            for k in range(numtopics):
                for j in range(self.ndocs):
                    labels_pred.append(f'T_{k}')

            labels_true = list(df[df['Model'] == worst_model].Ground_Truth_Target)

            print(f'Worst performing model name: {worst_model}')
            print(f'Labels True: {labels_true}')
            print(f'Labels Pred: {labels_pred}')
            print(f'Rand score: {(rand_score(labels_true, labels_pred))}')
            print(f'Rand score: {(adjusted_rand_score(labels_true, labels_pred))}')
            print(f'Finished in {(time.process_time() - t):.1f} Seconds.')


    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    '''--------- NMF Model: Similarity between "TopNDocs" to "Topic" -------------------'''
    def nmf_doc_topic_sim(self):
        self.__class__.dfp_nmf = pd.DataFrame()
        self.__class__.df_nmf  = pd.DataFrame()
        num_models = 0
        t    = time.process_time()     
        data = []
        dt1  = []
        
        for i in range(1, self.max_ngram_range):
            for j in range(1, self.max_ngram_range):
                if j >= i:
                    ngram = (i, j)
                    tfidf_vect = TfidfVectorizer(max_df = 0.9,
                                                 min_df = 1,
                                                 ngram_range  = ngram,
                                                 max_features = 2000)

                    A_tfidf = tfidf_vect.fit_transform(self.strip_text)

                    self.__class__.doc_terms_nmf = tfidf_vect.get_feature_names()
                    
                    # Topic Model
                    nmf = NMF(n_components = self.ntopics, init='nndsvd').fit(A_tfidf)

                    try:
                        del topics2present
                    except NameError:
                        pass

                    topics2present = []
                    for _, topic in enumerate(nmf.components_):
                        top_terms = [self.__class__.doc_terms_nmf[i] for i in topic.argsort()[-1:-self.nterms-1:-1]]
                        topics2present.append(top_terms)

                    topic_results = nmf.transform(A_tfidf)
                    
                    top_docs_indx = topic_results.argsort(axis=0)[-self.ndocs-1:]
                    
                    print(f'Ngram_{ngram}.')
                    for k in range(self.ntopics):
                        
                        queryTopic = TreebankWordDetokenizer().detokenize(topics2present[k])
                        
                        doc1 = nlp(queryTopic)
                        
                        if len(self.targets) > 0 :
                            
                            tdo = [self.targets[x] for x in topic_results.argsort(axis=0)[::-1][:,k]]
                            
                            dr = [f'Model_{num_models}_{ngram}', k , tdo]
                       
                            dt1.append(dr)
    
                        for l in range(self.ndocs):

                            query_doc = self.strip_text[top_docs_indx[:, k][l]]
                            
                            doc2 = nlp(query_doc)
                            
                            similarities = doc1.similarity(doc2)
                            
                            print(f'Topic_{k} & Doc_{top_docs_indx[:, k][l]} : {similarities}.')
                            
                            if len(self.targets) > 0:
                                
                                data_row = [f'Model_{num_models}_{ngram}',
                                        k,
                                        f'Topic_{k}: {topics2present[k]}',
                                        f'Doc_{top_docs_indx[:,k][l]}: {self.orig_msg_content[top_docs_indx[:,k][l]]}',
                                        f'Doc_{top_docs_indx[:,k][l]}: {self.strip_text[top_docs_indx[:,k][l]]}',
                                        self.targets[top_docs_indx[:,k][l]],
                                        similarities]
                            else:

                                data_row = [f'Model_{num_models}_{ngram}',
                                            k,
                                            f'Topic_{k}: {topics2present[k]}',
                                            f'Doc_{top_docs_indx[:,k][l]}: {self.orig_msg_content[top_docs_indx[:,k][l]]}',
                                            f'Doc_{top_docs_indx[:,k][l]}: {self.strip_text[top_docs_indx[:,k][l]]}',
                                            similarities]
                            
                            data.append(data_row)

                    num_models += 1

        if len(self.targets) > 0:
                        
            self.__class__.df_nmf = pd.DataFrame(data, columns=['Model',
                                                                'Topic_num',
                                                                'Extracted_Topics',
                                                                'Relevant_Docs',
                                                                'Strip_text_of_doc',
                                                                'Ground_Truth_Target',
                                                                'Doc_Topic_Similarity'])
            
            self.__class__.all_doc_targets_nmf = pd.DataFrame(dt1, columns = ['Model',
                                                                            'Topic_num',
                                                                            'all_doc_targets'])
                                                                            
        else:
            self.__class__.df_nmf = pd.DataFrame(data, columns = ['Model',
                                                                'Topic_num',
                                                                'Extracted_Topics',
                                                                'Relevant_Docs',
                                                                'Strip_text_of_doc',
                                                                'Doc_Topic_Similarity'])
     
    
        # ------- Plot the model average similarities ------- #
        df = self.__class__.df_nmf
        df.Doc_Topic_Similarity = df.Doc_Topic_Similarity.astype('float')
        numtopics = min(len(df['Topic_num'].drop_duplicates()), self.ntopics)
        lst = []
        index = []
        for m in range(num_models):
            
            model_name = df.Model[m*numtopics*self.ndocs]
            
            modelm_sim = [df.Doc_Topic_Similarity[i]
                          for i in range(len(df.Doc_Topic_Similarity))
                          if df.Model[i] == model_name]

            lst.append(statistics.mean(modelm_sim))
            index.append(model_name)
            print(f'Model Name: {model_name}: Average Sims: {statistics.mean(modelm_sim)}.')
            
            
        plt.bar(index, lst, width = 0.5)
        plt.xticks(rotation = 45)
        plt.margins(0.04)
        plt.ylabel("Similarity Index")
        plt.title("Doc Topic Similarity")
        
        # save and show
        plt.savefig(f'{self.chnlN}_NMF_DTS.png', dpi = 300, bbox_inches = 'tight')
        plt.show()
        
        
        # ------------
        df1  = pd.DataFrame(columns = ['Model_names', 'Average_Model_Sims'])
        df1['Model_names'] = index
        df1['Average_Model_Sims'] = lst
        lst = list(df1['Average_Model_Sims'])
        best_model  = df1['Model_names'][lst.index(max(lst))]
        worst_model = df1['Model_names'][lst.index(min(lst))]
        
        self.__class__.dfp_nmf['AverageAllModelSims'] = lst
        self.__class__.dfp_nmf['ModelNames'] = index
        
        
        if len(self.targets)>0:
            # ------------
            # For Best performing:
            # ------------
            df     = self.__class__.df_nmf
            dfad   = self.__class__.all_doc_targets_nmf
            target = list(set(self.targets))

            topic_nms = []

            numtopics = min(len(df['Topic_num'].drop_duplicates()),self.ntopics)

            for i in range(numtopics):
                topic_nms.append(f'T{i}')

            plotdata = {'Topics':topic_nms}

            for tar in target:
                plotdata[f'{tar}'] = [0]*numtopics 

            # --------- Rand Score, for all produced models
            modelname = list(df['Model'].drop_duplicates())
            rs = []
            labels_pred = []
            for k in range(numtopics):
                for j in range(self.ndocs):
                    labels_pred.append(f'T_{k}')

            for modelname in modelname:
                labels_true = list(df[df['Model'] == modelname].Ground_Truth_Target)
                rs.append(rand_score(labels_true, labels_pred))
                print(f'LDA: {modelname}. Rand Score:{rand_score(labels_true, labels_pred)}')    
            self.__class__.dfp_nmf['RandScoreAllModels'] = rs

            # ---------- plot best_performing - Rand Score included ----------

            for k in range(numtopics):
                
                list_k = list(df['Ground_Truth_Target'].loc[(df['Model'] == best_model) & (df['Topic_num'] == k)])

                for j in range(len(target)):

                    cnt = list_k.count(target[j])

                    plotdata[list(plotdata.keys())[j+1]][k] = cnt

            df_plot = pd.DataFrame(data = plotdata)
            df_plot.plot.bar(x = "Topics" , y = list(plotdata.keys())[1:], rot = 0, title = f'The Clustering Performance from \n NMF Best Performing Selected Model as {best_model}.')
            plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            plt.savefig(f'{self.chnlN}_NMFClusterBestModel_{best_model}.png')
            plt.show(block = True) 

            labels_true = list(df[df['Model'] == best_model].Ground_Truth_Target)

            print(f'Best performing model name: {best_model}')
            print(f'Labels True: {labels_true}')
            print(f'Labels Pred: {labels_pred}')
            print(f'Rand score: {(rand_score(labels_true, labels_pred))}')
            print(f'Rand score: {(adjusted_rand_score(labels_true, labels_pred))}')

        # ---------- plot worst_performing - Rand Score included -----------
            for k in range(numtopics):

                list_k = list(df['Ground_Truth_Target'].loc[(df['Model'] == worst_model) & (df['Topic_num'] == k)])

                for j in range(len(target)):

                    cnt = list_k.count(target[j])

                    plotdata[list(plotdata.keys())[j+1]][k] = cnt

            df_plot = pd.DataFrame(data = plotdata)
            df_plot.plot.bar(x = "Topics" , y = list(plotdata.keys())[1:], rot = 0, title = f'The Clustering Performance from \n NMF Worst Performing Selected Model as {worst_model}.')
            plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
            plt.savefig(f'{self.chnlN}_NMFClusterWorstModel_{worst_model}.png')
            plt.show(block = True) 

            labels_pred = []
            for k in range(numtopics):
                for j in range(self.ndocs):
                    labels_pred.append(f'T_{k}')

            labels_true = list(df[df['Model'] == worst_model].Ground_Truth_Target)

            print(f'Worst performing model name: {worst_model}')
            print(f'Labels True: {labels_true}')
            print(f'Labels Pred: {labels_pred}')
            print(f'Rand score: {(rand_score(labels_true, labels_pred))}')
            print(f'Rand score: {(adjusted_rand_score(labels_true, labels_pred))}')
            print(f'Finished in {(time.process_time() - t):.1f} Seconds.')

    
    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    ''' ------- NTM: Neural Doc-Topic-Sim Analysis ------- '''  
    def ntm_doc_topic_sim(self):
        # ---------------------------
        # ------------ CombinedTM - NTM
        qt = TopicModelDataPreparation("all-mpnet-base-v2")
        training_dataset = qt.fit(text_for_contextual = self.orig_msg_content, # csorig_corp, 
                                  text_for_bow = self.strip_text)

        ctm_ProdLDA = CombinedTM(bow_size   = len(qt.vocab), 
                         model_type = 'prodLDA', 
                         contextual_size = 768, 
                         n_components = self.ntopics, 
                         num_epochs = 15)        
        
        ctm_ProdLDA.fit(training_dataset)
        testing_dataset = qt.transform(text_for_contextual = self.orig_msg_content, 
                                       text_for_bow = self.strip_text)
        
        doc_topic_dist = ctm_ProdLDA.get_doc_topic_distribution(testing_dataset, n_samples=10) 
        
        tpcs = ctm_ProdLDA.get_topics()
        dr = []
        labels_true = []
        similarities = []
        for i in range(self.ntopics):
            
            queryTopic = TreebankWordDetokenizer().detokenize(tpcs[i])
            tpc_nlp = nlp(queryTopic)
            topdocs = ctm_ProdLDA.get_top_documents_per_topic_id(unpreprocessed_corpus = self.orig_msg_content, 
                                                         topic_id = i, 
                                                         document_topic_distributions = doc_topic_dist, 
                                                         k = self.ndocs)
            
            for j in range(self.ndocs):
                
                query_doc = topdocs[j][0]
                doc_nlp = nlp(query_doc)
                sims = tpc_nlp.similarity(doc_nlp) 
                similarities.append(sims)
                # get index of doc
                idx = self.orig_msg_content.index(query_doc)
                dr.append([f'CombinedTM_NTM',
                           i,
                          f'{tpcs[i]}',
                          f'Doc_{idx}: {query_doc}', 
                          sims])
                if len(self.targets)>0:
                    labels_true.append(self.targets[idx])

                
        # mean sims
        avg_simsC = statistics.mean(similarities)
        if len(self.targets)>0:
            # Rand-Score: find true and pred labels for rand-score
            labels_pred = []
            for k in range(self.ntopics):
                for j in range(self.ndocs):
                    labels_pred.append(f'T_{k}')

            rs_combinedTM = rand_score(labels_true, labels_pred)        
        
        # ---------------------------
        # ------------ ZeroShot - NTM
        del qt, training_dataset, testing_dataset, doc_topic_dist 
        qt = TopicModelDataPreparation("all-mpnet-base-v2")
        training_dataset = qt.fit(text_for_contextual = self.orig_msg_content, 
                                  text_for_bow = self.strip_text)
        
        ctm_zeroShot = ZeroShotTM(bow_size       = len(qt.vocab), 
                                 contextual_size = 768, 
                                 n_components    = self.ntopics,
                                 num_epochs      = 15)
        
        ctm_zeroShot.fit(training_dataset)
        
        testing_dataset = qt.transform(text_for_contextual = self.orig_msg_content, 
                                       text_for_bow        = self.strip_text)
        
        doc_topic_dist = ctm_zeroShot.get_doc_topic_distribution(testing_dataset, n_samples=10) 
        
        tpcs = ctm_zeroShot.get_topics()

        labels_true = []
        similarities = []
        for i in range(self.ntopics): 
            queryTopic = TreebankWordDetokenizer().detokenize(tpcs[i])
            tpc_nlp    = nlp(queryTopic)
            topdocs    = ctm_zeroShot.get_top_documents_per_topic_id(unpreprocessed_corpus = self.orig_msg_content, 
                                                           topic_id = i, 
                                                           document_topic_distributions = doc_topic_dist, 
                                                           k = self.ndocs)
            for j in range(self.ndocs):
                query_doc = topdocs[j][0]
                doc_nlp = nlp(query_doc)
                sims = tpc_nlp.similarity(doc_nlp) 
                similarities.append(sims)
                # get index of doc
                idx = self.orig_msg_content.index(query_doc)
                
                dr.append([f'ZeroShotTM_NTM',
                           i,
                          f'{tpcs[i]}',
                          f'Doc_{idx}: {query_doc}', 
                          sims])
                
                if len(self.targets) > 0:
                    labels_true.append(self.targets[idx])
        
        
        self.__class__.df_ntm = pd.DataFrame(dr, columns=['Model',
                                                          'Topic_num',
                                                          'Extracted_Topics',
                                                          'Relevant_Docs',
                                                          'Doc_Topic_Similarity'])
        
        avg_simsZ = statistics.mean(similarities)
         # Average Similarities
        self.__class__.avg_sims_ntm = [avg_simsC, avg_simsZ]
        
        
        # Rand-Scores: 
        if len(self.targets)>0:
            
            rs_zeroshot = rand_score(labels_true, labels_pred)
            
            self.__class__.rs_ntm = [rs_combinedTM, rs_zeroshot]
        
    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    ''' ------- Save Topics and Documents for further investigation. -------- '''    
    def save_Top_Doc(self, targetmodel, tmodel): 
        
        if tmodel   == 'lda':
            df = self.__class__.df_lda
        
        elif tmodel == 'nmf':
            df = self.__class__.df_nmf
            
        elif tmodel == 'ntm':
            df = self.__class__.df_ntm

        tps = list(df['Extracted_Topics'].loc[df['Model'] == targetmodel])

        docs = list(df['Relevant_Docs'].loc[df['Model'] == targetmodel])
        
        f = open(f'{self.chnlN}_{targetmodel}_{tmodel}.txt', "w", encoding='utf-8')
        f.write(f'This document contains the top {self.ntopics} topics extracted from {self.chnlN}. Each topic is followed by the {self.ndocs} top corresponding Documents in the corpus.\n\n')
        
        f.write('--------------------------------------------------------------------------------- \n\n')
        
        for i in range(self.ntopics):
            f.write(f'{tps[self.ndocs*i]} \n\n')
            
            for j in range(self.ndocs):
                f.write(f'{docs[self.ndocs*i+j]} \n\n')
                
            f.write('--------------------------------------------------------------------------------- \n\n')
                
        f.close() 
        
        
    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    '''---------------------------------------------------------------------------------'''
    def plot_DTS_RS(self):
        
        # This function should be called after calling the lda and nmf functions. 
        # Document Topic Similarity
        dts_avg_lda = list(self.__class__.dfp_lda['AverageAllModelSims'])
        dts_avg_nmf = list(self.__class__.dfp_nmf['AverageAllModelSims'])
        dts_avg_ntm = self.__class__.avg_sims_ntm        
        DTS = [*dts_avg_lda,*dts_avg_nmf, *dts_avg_ntm] 

        # Rand Score
        rs_lda  = list(self.__class__.dfp_lda['RandScoreAllModels'])
        rs_nmf  = list(self.__class__.dfp_nmf['RandScoreAllModels'])
        rs_ntm  = self.__class__.rs_ntm
        RS = [*rs_lda , *rs_nmf , *rs_ntm]

        labels_lda = []
        labels_nmf = []
        labels_ntm = ['NTM_CombinedTM','NTM_ZeroShot'] 
        for i in range(len(dts_avg_lda)):
            labels_lda.append(f'L{i}')
            labels_nmf.append(f'N{i}')
        
        labels = [*labels_lda, *labels_nmf, *labels_ntm]
        
        plt.scatter(x = RS, y = DTS)
        for i in range(len(DTS)):
            plt.text(RS[i] + 0.01, DTS[i] + 0.01, labels[i])
        
        plt.title("DTS averaged for each model, Rand Score is measured for each model. ")
        plt.xlabel("RS")
        plt.ylabel("DTS")
        plt.xlim(min(RS) - 0.05, max(RS) + 0.05)
        plt.ylim(min(DTS) - 0.05, max(DTS) + 0.05)
        
        plt.savefig(f'{self.chnlN}_DTS_RS_{self.ndocs}.png', dpi = 300, bbox_inches = 'tight')
        plt.show()


'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
def main():

    dts = DocTopicSim(chnlN = 'ChannelName',
                      orig_msg_content = origCorp,
                      strip_text = cleanedCorp,
                      max_ngram_range = 5,
                      targets = [],
                      ntopics = 10,
                      nterms = 10,
                      ndocs = 5)
    
    dts.lda_doc_topic_sim()
    df = dts.df_lda
    df.to_csv('chName10topic_lda.csv', index=False)

    dts.nmf_doc_topic_sim()
    df = dts.df_nmf
    df.to_csv('chName10topic_nmf.csv', index=False)

    dts.ntm_doc_topic_sim()
    df = dts.df_ntm
    df.to_csv('chName10topic_ntm.csv', index=False)

    if len(targets) > 0 :
        dts.plot_DTS_RS()

''' ------------------------------------------------------------------------'''
if __name__ == "__main__":
    main()  
