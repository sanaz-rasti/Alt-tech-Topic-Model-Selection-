''' --------------------------- Doc Topic Similarity Class --------------------------- '''
'''
DESCRIPTION: 
    DocTopicSim class is created to perform ... major topic modeling experiments as follows:
        - Models TF-IDF topic modeling for a different range of n-grams 
        - Detects N-Top-Related-Documents for each extracted topic
        - Calculates similarity between each pair of "N-Top-Rselated-Document" and "Corresponding-Topic"
        - Returns a descriptive DataFrame of corpus include columns = ['Model',
                                                                     'Num_Topics', 
                                                                     'Extracted_Topics',
                                                                     'Relevant_Docs',
                                                                     'Strip_text_of_doc',
                                                                     'Doc_Topic_Similarity'] 
        - Plot the average Ndoc-topic similarity for each experimented model. 
        
    
INPUT:
    The input for the class requires:
                    - orig_msg_content
                    - strip_text
                    - ngram_range
                    - StopWords text file
                    - Number of topics: ntopics
                    - Number of terms per topic: nterms
                    - Number of relevant documnets to be assigned from each extracted topic: ndocs
OUTPUT:
        - Calling tfidf_doc_topic_sim() function on DocTopicSim-class-object will store a DataFrame in 
            class.df variable which includes the following content in columns: ['Model',
                                                                                'Num_Topics',
                                                                                'Extracted_Topics',
                                                                                'Relevant_Docs',
                                                                                'Strip_text_of_doc',
                                                                                'Doc_Topic_Similarity']
                                                                                
        - Calling plot_similarity_for_models() function on DocTopicSim-class-object which is trained using 
            tfidf_doc_topic_sim() will plot the average similarity which is achieved for each trained model. 
                                                                            
                                                                            
        
EXAMPLE:
    Employing the ChannelData class, strip_text and orig_msg_content will be achieved 
    chd = ChannellData(datapath=PATH-TO-DATABASE, 
                   target_table=TARGETED-TABLE, 
                   channelUniqueId=CHANNEL-ID)
    strip_text, orig_msg_content = chd.corpus()
    
    Create object of DocTopicSim with ngram_range=5,ntopics=10, ndocs=3
    dts = DocTopicSim(orig_msg_content= orig_msg_content, 
                  strip_text = strip_text, 
                  ngram_range = 5, 
                  StopWords_txtfile = 'stopwords.txt', 
                  ntopics = 10, 
                  nterms = 10,
                  ndocs = 3)
    dts.tfidf_doc_topic_sim()
    dts.plot_similarity_for_models()
'''
''' ---------------------------  Doc Topic Similarity --------------------------- '''


class DocTopicSim:

    doc_terms = []
    StopWords = stopwords.words('english')
    num_models = 0
    df = pd.DataFrame()

    def __init__(self,
                 orig_msg_content: 'Should take the original meseg content from the database',
                 strip_text: 'Strip text data from database',
                 ngram_range,
                 StopWords_txtfile: 'The stopwords text file',
                 ntopics: 'Number of desired topics to be extracted',
                 nterms: 'Number of terms per topic',
                 ndocs: 'Number of documents to be extracted') -> None:

        self.orig_msg_content = orig_msg_content
        self.strip_text = strip_text
        self.ngram_range = ngram_range
        self.__class__.StopWords.extend(list(set([str(i for i in range(0, 100000))] +
                                                 list(np.loadtxt(StopWords_txtfile, dtype='str')))))
        self.ntopics = ntopics
        self.nterms = nterms
        self.ndocs = ndocs
        ''' Gensim Dictionary Prepration'''
        text = []
        for txt in strip_text:
            text.append(txt.split())

        self.dictionary_for_gensim = corpora.Dictionary(text)
        self.index_tmpfile = get_tmpfile("indxtempfile")
        ''' Random state for nmf '''
        random_state = np.random.randint(len(strip_text), 1000, size=1)
        self.rndState = random_state[0]

    '''------------------------------ Similarity between "TopNDocs" to "Topic" ------------------------------'''

    def tfidf_doc_topic_sim(self):

        data = []
        for i in range(1, self.ngram_range):
            for j in range(1, self.ngram_range):
                if j >= i:

                    # print(f'Model_{self.num_models}')
                    ngram = (i, j)

                    #print(f'Experimenting Doc-Topic Similarities for Model_{self.num_models} with NgramRange={ngram}:...')
                    tfidf_vect = TfidfVectorizer(max_df=0.9,
                                                 min_df=2,
                                                 ngram_range=ngram,
                                                 token_pattern=r'\b[^\d\W]+\b',
                                                 max_features=5000,
                                                 stop_words=self.__class__.StopWords)

                    A_tfidf = tfidf_vect.fit_transform(self.strip_text)

                    self.__class__.doc_terms = tfidf_vect.get_feature_names()

                    nmf = NMF(n_components=self.ntopics,
                              random_state=self.rndState, init='nndsvd').fit(A_tfidf)

                    try:
                        del topics2present
                    except NameError:
                        pass

                    topics2present = []
                    for _, topic in enumerate(nmf.components_):
                        top_terms = [self.__class__.doc_terms[i]
                                     for i in topic.argsort()[-1:-self.nterms-1:-1]]
                        topics2present.append(top_terms)

                    topic_results = nmf.transform(A_tfidf)
                    top_docs_indx = topic_results.argsort(
                        axis=0)[-self.ndocs-1:-1]
                    #print(f'Topics to present for model_{self.num_models}: {topics2present}')

                    for k in range(self.ntopics):
                        for l in range(self.ndocs):
                            # find similarity between pairs of:
                            # 'self.msg_content[top_docs_indx[:,k][self.ndocs-l]]' and 'topics2present[k]'

                            query_doc = [self.dictionary_for_gensim.doc2bow(txt)
                                         for txt in [nltk.word_tokenize(self.orig_msg_content[top_docs_indx[:, k][l]])]]

                            index = Similarity(
                                self.index_tmpfile, query_doc, num_features=len(self.dictionary_for_gensim))

                            # class splits the index into several smaller sub-indexes ("shards")
                            # tokenize topics2present[k]
                            tokenizedtopicK = nltk.word_tokenize(
                                ' '.join(topics2present[k]))

                            query_topic = self.dictionary_for_gensim.doc2bow(
                                tokenizedtopicK)

                            #print(f'the query topic for k={k}:{query_topic}.')

                            similarities = index[query_topic]

                            data_row = [f'Model_{self.num_models}_{ngram}',
                                        self.ntopics,
                                        f'Topic_{k}: {topics2present[k]}',
                                        f'Doc_{top_docs_indx[:,k][l]}: {self.orig_msg_content[top_docs_indx[:,k][l]]}',
                                        f'Doc_{top_docs_indx[:,k][l]}: {self.strip_text[top_docs_indx[:,k][l]]}',
                                        similarities]
                            data.append(data_row)

                    self.num_models += 1
        self.__class__.df = pd.DataFrame(data, columns=['Model',
                                                        'Num_Topics',
                                                        'Extracted_Topics',
                                                        'Relevant_Docs',
                                                        'Strip_text_of_doc',
                                                        'Doc_Topic_Similarity'])

    ''' Plot the model average similarities '''

    def plot_similarity_for_models(self):

        self.__class__.df.Doc_Topic_Similarity = self.__class__.df.Doc_Topic_Similarity.astype(
            'float')

        lst = []

        index = []

        for m in range(self.num_models):
            model_name = self.__class__.df.Model[m*self.ntopics*self.ndocs]
            modelm_sim = [self.__class__.df.Doc_Topic_Similarity[i]
                          for i in range(len(self.__class__.df.Doc_Topic_Similarity))
                          if self.__class__.df.Model[i] == model_name]

            lst.append(statistics.mean(modelm_sim))

            index.append(model_name)

        plot_data = pd.DataFrame({"Average Doc-Topic Sim": lst}, index=index)

        plot_data.plot(kind="bar")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Gensim Cosine Similarity')
        plt.savefig('Models_doc_topic_sim.png', dpi=300, bbox_inches='tight')
        plt.show()


'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------'''