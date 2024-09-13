import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
pd.set_option('display.max_columns', None)

# reading the data
comments = pd.read_csv('/Users/gladys/Documents/GitHub/SMA_A02/A02_tiktok/EDA/eda_data/comments_tt.csv')
print(comments)

# setting up the data
# converting data to date variable
comments['createTimeISO'] = pd.to_datetime(comments['createTimeISO']) # convert
comments['createTimeISO'] = comments['createTimeISO'].dt.date # extract date only

# SENTIMENT SCORES ------------------------------------------
# getting sentiment score
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

comments['Sentiment'] = comments['text'].apply(get_sentiment)

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

comments['Sentiment_category'] = comments['Sentiment'].apply(categorize_sentiment)

print(comments)

# SENTIMENT SCORE OVERTIME ANALYSIS
comments['createTimeISO'] = pd.to_datetime(comments['createTimeISO'])
monthly_avg_sentiment = comments.set_index('createTimeISO').resample('M')['Sentiment'].mean()

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

plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/tiktok/comments_lineplot.png', format='png', bbox_inches='tight', dpi=1200)

# SENTIMENT VS LIKES
likes_by_sentiment = comments.groupby('Sentiment_category')['Likes'].sum().reindex(
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

plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/tiktok/sentilikesplot.png', format='png', bbox_inches='tight', dpi=1200)


# SENTIMENT CATEGORIZATION PROPORTION
comments['Sentiment_category'] = comments['Sentiment'].apply(categorize_sentiment)
total_by_sentiment = comments['Sentiment_category'].value_counts().reindex(
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

plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/tiktok/senticomments.png', format='png', bbox_inches='tight', dpi=1200)

# WORD CLOUD
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size < 50:
        return '#DDE0E4'  
    elif font_size > 90:
        return '#8198B6'
    else:
        return '#B4C9E2' 
    
sentiments = comments['Sentiment_category'].unique()

for sentiment in sentiments:
    text_data = ' '.join(comments[comments['Sentiment_category'] == sentiment]['text'])

    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=custom_color_func).generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment}', pad=20)
    
    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/senti_figures/tiktok/wordcloud/{sentiment.replace(' ', '_')}_wordcloud.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()

# saving to csv
comments.to_csv('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Sentiment Analysis/data/senti_tt.csv')
