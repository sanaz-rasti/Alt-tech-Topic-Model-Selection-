
def corpus_tokens(corpus):
    count = 0 
    for doc in corpus:
        count += len(doc.split())
    return count 

'''-------------------------------------'''
def memoize(func):
    # storing results in cache 
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            
        return cache[key]
    
    return wrapper



@memoize
def data_cleaning(corpus):
    
    StopWords = stopwords.words('english')
    
    StopWords.extend(list(np.loadtxt('stopwords.txt', dtype='str')))
    
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
    
    corpus_copy = corpus
    
    # looking for clean_corp
    clean_corp = []
    for txt in corpus:
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
        
        if txt is '':
            corpus_copy.remove(text)
        else:
            clean_corp.append(txt)
        
        clean_corp  = list(filter(None, clean_corp))
        corpus_copy = list(filter(None, corpus_copy))
    
    return corpus_copy, clean_corp
