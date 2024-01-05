import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('vader_lexicon')
nltk.download('stopwords')


text_data = tweets['text']

sid = SentimentIntensityAnalyzer()
tweets['compound'] = text_data.apply(lambda x: sid.polarity_scores(x)['compound'])

plt.figure(figsize=(10, 6))
tweets['compound'].hist(bins=30, edgecolor='black')
plt.title('Sentiment Distribution')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Number of Tweets')
plt.show()

positive_tweets = ' '.join(text_data[tweets['compound'] > 0.2])
negative_tweets = ' '.join(text_data[tweets['compound'] < -0.2])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate(negative_tweets)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Positive Word Cloud')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Negative Word Cloud')
plt.axis('off')

plt.show()
