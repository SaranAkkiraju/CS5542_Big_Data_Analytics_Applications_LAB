import csv
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import matplotlib.pyplot as plot
plot.rcdefaults()
import numpy as np
import time

start = int(round(time.time() * 1000))

keywords = ['yellow', 'red', 'stop', 'road','traffic']
punctuations = "?:!.,;"
stop_words = set(stopwords.words('english'))

with open(r'C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Lab1\dataset\SBU_captioned_photo_dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv_reader)
    captions = []
    urls =[]
    my_captions = []
    my_urls = []
    word0_count = 0
    word1_count = 0
    word2_count = 0
    word3_count = 0
    word4_count = 0
    
    # Splittig colums of CSV data into captions and Urls
    for index in range(1,len(data)):
        captions.append(data[index][0])
        urls.append(data[index][1])
        
    for new_index  in range(0, len(captions)):
        
        # finding tokens in a caption
        tokens = word_tokenize(captions[new_index])
        
        # Removing punctuations and stop_words from tokens
        for word in tokens:
            if word in punctuations:
                tokens.remove(word)
            if word in stop_words:
                tokens.remove(word)
        
        # Performing lemmetization
        distinct_tokens = list(set(tokens)) 

        for word in distinct_tokens:
            lemmit_word = wordnet_lemmatizer.lemmatize(word, pos="v")       
            if lemmit_word in keywords:
                my_captions.append(captions[new_index])
                my_urls.append(urls[new_index])
                if lemmit_word == keywords[0]:
                   word0_count+=1
                if lemmit_word == keywords[1]:
                   word1_count+=1
                if lemmit_word == keywords[2]:
                   word2_count+=1
                if lemmit_word == keywords[3]:
                   word3_count+=1
                if lemmit_word == keywords[4]:
                   word4_count+=1
# Pushing the consolidated dataset into new csvfile            
with open('NewDataSet.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerows(zip(my_captions, my_urls))

# Showing the Keyword count     
y_pos = np.arange(len(keywords))
performance = [word0_count,word1_count,word2_count,word3_count,word4_count]
plot.bar(y_pos, performance, align='center', alpha=0.5)
plot.xticks(y_pos, keywords)
plot.show()
end = int(round(time.time() * 1000))
print("Time for building convnet: ")
print(end - start)


    