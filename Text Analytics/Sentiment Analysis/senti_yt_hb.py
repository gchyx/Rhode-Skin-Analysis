import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
pd.set_option('display.max_columns', None)

# reading the data
hailey_b = pd.read_csv('/Users/gladys/Documents/GitHub/SMA_A02/A02_yt/EDA/eda_data/haileyb_eda.csv')
print(hailey_b)

# setting up the data
# converting data to date variable
hailey_b['Timestamp'] = pd.to_datetime(hailey_b['Timestamp']) # convert
hailey_b['Timestamp'] = hailey_b['Timestamp'].dt.date # extract date only


# SENTIMENT SCORES ------------------------------------------
# getting sentiment score
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

hailey_b['Sentiment'] = hailey_b['Comment'].apply(get_sentiment)

# categorizing the sentiment scores
def categorize_sentiment(score):
    if score < -0.5:
        return 'Strongly Negative'
    elif -0.5 <= score < -0.1:
        return 'Negative'
    elif -0.1 <= score < 0.1:
        return 'Neutral'
    elif 0.1 <= score < 0.5:
        return 'Positive'
    else:
        return 'Strongly Positive'

hailey_b['Sentiment_category'] = hailey_b['Sentiment'].apply(categorize_sentiment)

print(hailey_b)
''''
# SENTIMENT SCORE OVERTIME ANALYSIS
hailey_b['Timestamp'] = pd.to_datetime(hailey_b['Timestamp'])
monthly_avg_sentiment = hailey_b.set_index('Timestamp').resample('M')['Sentiment'].mean()

# The line plot
plt.figure(figsize=(12, 8))
plt.plot(monthly_avg_sentiment.index, monthly_avg_sentiment, color='#B4C9E2')
plt.title('Average Sentiment Score Over Time', pad=15)
plt.xlabel('Date', labelpad=15)
plt.ylabel('Average Sentiment Score', labelpad=15)
plt.xticks(rotation=0) 
plt.tight_layout() 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(bottom=-0.2)

plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/youtube/hailey/hb_lineplot.png', format='png', bbox_inches='tight', dpi=1200)

# SENTIMENT VS LIKES
likes_by_sentiment = hailey_b.groupby('Sentiment_category')['Likes'].sum().reindex(
    ['Strongly Negative', 'Negative', 'Neutral', 'Positive', 'Strongly Positive'])

# The bar plot
plt.figure(figsize=(12, 8))
likes_by_sentiment.plot(kind='bar', color='#B4C9E2', edgecolor='#8198B6')
plt.title('Total Likes by Sentiment Category', pad=20)
plt.xlabel('Sentiment Category', labelpad=15)
plt.ylabel('Total Likes', labelpad=15)
plt.xticks(rotation=0) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/youtube/hailey/hb_sentilikesplot.png', format='png', bbox_inches='tight', dpi=1200)


# SENTIMENT CATEGORIZATION PROPORTION
hailey_b['Sentiment_category'] = hailey_b['Sentiment'].apply(categorize_sentiment)
total_by_sentiment = hailey_b['Sentiment_category'].value_counts().reindex(
    ['Strongly Negative', 'Negative', 'Neutral', 'Positive', 'Strongly Positive'])
proportion_by_sentiment = total_by_sentiment / total_by_sentiment.sum()

# The bar plot
plt.figure(figsize=(12, 8))
total_by_sentiment.plot(kind='bar', color='#B4C9E2', edgecolor='#8198B6')
plt.title('Total Number of Comments by Sentiment Category', pad=20)
plt.xlabel('Sentiment Category', labelpad=15)
plt.ylabel('Total Number of Comments', labelpad=15)
plt.xticks(rotation=0) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/youtube/hailey/hb_senticomments.png', format='png', bbox_inches='tight', dpi=1200)
'''
# WORD CLOUD
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size < 50:
        return '#DDE0E4'  
    elif font_size > 90:
        return '#8198B6'
    else:
        return '#B4C9E2' 
    
sentiments = hailey_b['Sentiment_category'].unique()

for sentiment in sentiments:
    text_data = ' '.join(hailey_b[hailey_b['Sentiment_category'] == sentiment]['Comment'])

    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=custom_color_func).generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment}', pad=20)
    
    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/youtube/wordcloud/{sentiment.replace(' ', '_')}_wordcloud_hb.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()

# saving to csv
hailey_b.to_csv('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/data/senti_yt_hb.csv')