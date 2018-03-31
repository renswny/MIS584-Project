import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def process_text(text_lst):
    res = []
    regex = re.compile('[^a-zA-Z ]')
    print len(text_lst)
    for text in text_lst:
        text = str(text).lower()
        text = regex.sub('', text) #remove non-alphabet characters
        #remove stop words:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [w for w in word_tokens if not w in stop_words]
        res.extend(filtered_text)
        # res.append(' '.join(filtered_text))
    return ' '.join(res)

reviews = pd.read_csv('data/reviews.csv', sep=',')
print ('Total reviews: ', reviews.shape[0])
#print reviews.head()

#take the neccessary columns:
column_names = ['reviewerID', 'asin', 'reviewText', 'overall', 'reviewTime']
reviews = reviews[column_names]
#print reviews.head()

#take the bad reviews?
bad_reviews = reviews[reviews['overall'] < 4]
print ('Total bad reviews:', bad_reviews.shape[0])
bad_review_txts = bad_reviews[['reviewText']].values

# print bad_reviews.ix[1, 'reviewText']
# for i in range(bad_reviews.shape[0]):
#     bad_reviews.ix[i, ['reviewText']] = process_text(bad_reviews.loc[i, ['reviewText']])
# print bad_reviews.loc[0, ['reviewText']]

print len(bad_review_txts)
bad_review_txts =  process_text(bad_review_txts)
print len(bad_review_txts)


#drawing word cloud:
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

guitar_mask = np.array(Image.open("data/guitar.jpg"))
stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white", max_words=2000, mask=guitar_mask
               )
# generate word cloud
wc.generate(bad_review_txts)

# store to file
wc.to_file("data/alice.png")

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
# plt.figure()
# plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
# plt.axis("off")
plt.show()


