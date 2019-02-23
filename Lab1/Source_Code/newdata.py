import csv
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
import matplotlib.pyplot as plot
plot.rcdefaults()
import numpy as np
import time
import random
import urllib.request
import os

start = int(round(time.time() * 1000))

keywords = ['yellow', 'red', 'stop', 'road','traffic']
punctuations = "?:!.,;"
stop_words = set(stopwords.words('english'))

dataset_path = r'C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Lab1\dataset'
with open(r'C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Lab1\dataset\SBU_captioned_photo_dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    data = list(csv_reader)
    captions = []
    urls =[]
    my_captions = []
    my_urls = []
    # Arrays for storing sample images based on Keywords
    yellow = []
    red = []
    stop = []
    road = []
    traffic = []
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
                   yellow.append(urls[new_index])
                if lemmit_word == keywords[1]:
                   word1_count+=1
                   red.append(urls[new_index])
                if lemmit_word == keywords[2]:
                   word2_count+=1
                   stop.append(urls[new_index])
                if lemmit_word == keywords[3]:
                   word3_count+=1
                   road.append(urls[new_index])
                if lemmit_word == keywords[4]:
                   word4_count+=1
                   traffic.append(urls[new_index])
                   
# Pushing the consolidated dataset into new csvfile            
with open('NewDataSet.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    writer.writerows(zip(my_captions, my_urls))
    
# Logic for downloading an image from url    
def download_image(url,length,myPath):
    name = random.randrange(1,length)
    filename = str(name)+".jpg"
    fullfilename = os.path.join(myPath, filename)
    urllib.request.urlretrieve(url,fullfilename)     

# adding path for new folders of Keywords
yellow_path  = os.path.join(dataset_path, 'yellow')
red_path     = os.path.join(dataset_path, 'red')
stop_path    = os.path.join(dataset_path, 'stop')
road_path    = os.path.join(dataset_path, 'road')
traffic_path = os.path.join(dataset_path, 'traffic')

yellow_length = len(yellow) 
red_length = len(red)
stop_length = len(stop)
road_length = len(road)
traffic_length = len(traffic)
   
# Sampling Images based on Key_words
for value in yellow:
    download_image(value,yellow_length,yellow_path)
    
for value in red:
    download_image(value,red_length,red_path)

for value in stop:
    download_image(value,stop_length,stop_path)

for value in road:
    download_image(value,road_length,road_path)

for value in traffic:
    download_image(value,traffic_length,traffic_path)    
        
# Showing the Keyword count     
y_pos = np.arange(len(keywords))
performance = [word0_count,word1_count,word2_count,word3_count,word4_count]
plot.bar(y_pos, performance, align='center', alpha=0.5)
plot.xticks(y_pos, keywords)
plot.show()
end = int(round(time.time() * 1000))
print("Time for building convnet: ")
print(end - start)


    