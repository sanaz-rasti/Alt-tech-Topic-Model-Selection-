

# Description

The original text corpus of our project is preprocessed using the script data_preprocessing.py. 

### ---------------------------------------------
### 
The script includes data_cleaning class for reading the corpus from a .txt file and preprocess. The class description is included in the following.  

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


<figure>
<center>
  <img src="./imgs/dc.png" alt="data_cleaning" style="width:70%">
  <figcaption>The pseudocode of text preprocessing. </figcaption>
  </center>
</figure>



The following text example presents an example of original message content and stripped text; the test sample is taken from Telegram channel Computing Forever

<figure>
<center>
  <img src="./imgs/dps.png" alt="preprocesing_example" style="width:70%">
  <figcaption>Sample of data preprocessing from Telegram channel Computing Forever. </figcaption>
  </center>
</figure>


The customized list of stop-words can be found in 'stopwords.txt' file; The categorized list of stopwords is presented in the following tables: 
<figure>
<center>
  <img src="./imgs/0.png" alt="" style="width:85%">
  </center>
</figure>

<figure>
<center>
  <img src="./imgs/1.png" alt="" style="width:70%">
  </center>
</figure>

<figure>
<center>
  <img src="./imgs/2.png" alt="" style="width:70%">
  </center>
</figure>
<figure>
<center>
  <img src="./imgs/3.png" alt="" style="width:70%">
  </center>
</figure>
<figure>
<center>
  <img src="./imgs/4.png" alt="" style="width:90%">
  </center>
</figure>
<figure>
<center>
  <img src="./imgs/5.png" alt="" style="width:90%">
  </center>
</figure>


### ---------------------------------------------
### How to Use the data_preprocessing.py  

- running the data_preprocessing script will ask for original corpus file and list of stopwords. 
The script will produce two .txt files, original_corpus.txt and clean_corp.txt files.
 
