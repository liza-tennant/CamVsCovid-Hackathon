# cam-vs-covid.py 
"""
This code takes in text responses for a single petition as a txt file, 
cleans the text data and outputs a wordcloud and topic analysis. 

""" 
import os
import json
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import pyLDAvis
import plotly.express as px
from sklearn.utils import resample
from operator import itemgetter
import seaborn as sns
import gensim
from nltk.stem import PorterStemmer,WordNetLemmatizer
import matplotlib.pyplot as plt

os.chdir('..')
os.listdir()
os.chdir('CamVsCovid')
#import file
with open('lockdown_tweets.txt', 'r') as f:
    text_data = f.read().splitlines() 
#descriptives 
print("\nNumber of responses in the dataset: {}".format(len(text_data)))


############# cleaning (preprocessing) text for frequency analysis  ############
text_list = []
## NB TEST THIS LATER 
for response in text_data:
    text = response['text']
    demographics = response['demographics'] #includes age, gender, county, occupation type? 
    response_type = response['type'] #
    text_list.append([text,demographics,response_type])

#test
text_list[0]

import nltk, re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

stop_words = stopwords.words('english')
normalizer = WordNetLemmatizer()

def get_part_of_speech(word):
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in probable_part_of_speech if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in probable_part_of_speech if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in probable_part_of_speech if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in probable_part_of_speech if item.pos()=="r"]  )
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    return most_likely_part_of_speech

def preprocess_text(text):
    cleaned = re.sub(r'\W+', ' ', text).lower()
    tokenized = word_tokenize(cleaned)
    normalized = [normalizer.lemmatize(token, get_part_of_speech(token)) for token in tokenized]
    return normalized

print((text_list[0])) #this hides the weird signs around the hashtag 

#apply the functions above to preprocess text 
processed_texts = [preprocess_text(text) for text in text_list]
processed_texts[:2] #this changes it from a sentence 'Lockdown has no ...' to a list of tokens, such as 'lockdown', 'have', 'no', etc.


# stop words removal
stop_words = set(stopwords.words('english'))

texts_no_stops = []
for text in processed_texts:
    text_no_stops = [word for word in text if word not in stop_words]
    texts_no_stops.append(text_no_stops)
texts_no_stops
#see notebook on github if we need to convert corpus to BoW format

# putting together the cleaned up words into one single string so that tf-idf can accept it

cleaned_sentences = []
for item in processed_texts:
    concat=''
    for word in item:
        concat+=word+' '
    cleaned_sentences.append(concat)
## NB MAYBE FEED THE texts_no_stops LIST TO TF-IDF INSTEAD 
cleaned_sentences[:2]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None)
tfidf_scores = vectorizer.fit_transform(cleaned_sentences)

print(type(tfidf_scores))
print(tfidf_scores.shape) #3, 101
tfidf_scores

# word embeddings
#nlp = spacy.load('en_core_web_md')
#texts_embeddings = np.array([nlp(text).vector for text in cleaned_sentences])
#titles_embeddings.shape



# checking the sentiment of the texts
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
text_sentiments = []
for text in cleaned_sentences:
    text_sentiments.append(sid.polarity_scores(text)['compound'])
text_sentiments = np.array(text_sentiments)
text_sentiments #array([0.5574, 0.4939, 0.1154])
#abstract_sentiments = abstract_sentiments.reshape((12387,1))


#### visualise wordclouds without LDA ####
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# text_data is our dataset of tweets

#if I need to group them by policy 
# policy = df.groupby("policy")

# Create and generate a word cloud image:

wordcloud1 = WordCloud(max_words=100, background_color="white").generate(text_data[0])
wordcloud2 = WordCloud(max_words=100, background_color="white").generate(text_data[1])
wordcloud3 = WordCloud(max_words=100, background_color="white").generate(text_data[2])



# Display the generated image:

plt.figure()
plt.imshow(wordcloud3, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud3')
plt.show()



######## for later - LDA for topic analysis of texts provided by users in free form #######
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(learning_method='online', n_components=10)
lda_tfidf = lda.fit_transform(tfidf_scores)

print("\n\n~~~ Topics found by tf-idf LDA ~~~")
for topic_id, topic in enumerate(lda.components_):
    message = "Topic #{}: ".format(topic_id + 1)
    message += " ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()])
    print(message)


from gensim import corpora, models

# list_of_list_of_tokens = [["a","b","c"], ["d","e","f"]]
# ["a","b","c"] are the tokens of document 1, ["d","e","f"] are the tokens of document 2...
dictionary_LDA = corpora.Dictionary(titles_no_stops)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in titles_no_stops]

num_topics = 20
%time lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary_LDA, passes=4, alpha=[0.01]*num_topics, eta=[0.01]*len(dictionary_LDA.keys()))
                                  
for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(str(i)+": "+ topic)
    print()





#### wordcloud of LDA topics ####
# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]+[color for name, color in mcolors.TABLEAU_COLORS.items()] # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(background_color='white', width=2500, height=1800, max_words=20, colormap='tab10', color_func=lambda *args, **kwargs: cols[i], prefer_horizontal=1.0)

topics = lda_model.show_topics(num_topics = 20, formatted=False)

fig, axes = plt.subplots(5, 4, figsize=(20,20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
