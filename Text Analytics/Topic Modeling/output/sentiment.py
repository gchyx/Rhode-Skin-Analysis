import matplotlib.pyplot as plt
import pandas as pd

# READING DATA
ig_comments = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/instagram/data/igcomment_topic.csv")
tt_comments = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/tiktok/data/ttcomment_topic.csv")
yt_comments = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/youtube/data/ytcomment_topic.csv")

# MINOR DATA PREPROCESSING ----------------------------
ig_comments['Sentiment_category'] = ig_comments['Sentiment_category'].replace({
    'Strongly Negative': 'Negative',
    'Strongly Positive': 'Positive'
})
tt_comments['Sentiment_category'] = ig_comments['Sentiment_category'].replace({
    'Strongly Negative': 'Negative',
    'Strongly Positive': 'Positive'
})
yt_comments['Sentiment_category'] = ig_comments['Sentiment_category'].replace({
    'Strongly Negative': 'Negative',
    'Strongly Positive': 'Positive'
})

# TOPIC'S SENTIMENT BAR GRAPH ----------------------------
# ig
topic_mapping = dict(zip(ig_comments['topic_id'], ig_comments['topic_name']))
sentiment_distribution = ig_comments.groupby(['topic_id', 'Sentiment_category']).size().unstack(fill_value=0)

for topic_id in sentiment_distribution.index:
    topic_name = topic_mapping[topic_id] 
    sentiment_counts = sentiment_distribution.loc[topic_id]
    max_sentiment = sentiment_counts.idxmax()
    colors = ['#8198B6' if sentiment == max_sentiment else '#DDE0E4' for sentiment in sentiment_counts.index]

    plt.figure(figsize=(10, 6))
    sentiment_distribution.loc[topic_id].plot(kind='bar', color=colors, alpha=0.7)
    plt.title(f'Sentiment Distribution for Topic {topic_id}: {topic_name}', pad=20)
    plt.xlabel('Sentiment Category', labelpad=15)
    plt.ylabel('Number of Mentions', labelpad=15)
    plt.xticks(rotation=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/sentiment/ig/distribution/dist_{topic_id}.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()

# tt
topic_mapping = dict(zip(tt_comments['topic_id'], tt_comments['topic_name']))
sentiment_distribution = tt_comments.groupby(['topic_id', 'Sentiment_category']).size().unstack(fill_value=0)

for topic_id in sentiment_distribution.index:
    topic_name = topic_mapping[topic_id] 
    sentiment_counts = sentiment_distribution.loc[topic_id]
    max_sentiment = sentiment_counts.idxmax()
    colors = ['#8198B6' if sentiment == max_sentiment else '#DDE0E4' for sentiment in sentiment_counts.index]

    plt.figure(figsize=(10, 6))
    sentiment_distribution.loc[topic_id].plot(kind='bar', color=colors, alpha=0.7)
    plt.title(f'Sentiment Distribution for Topic {topic_id}: {topic_name}', pad=20)
    plt.xlabel('Sentiment Category', labelpad=15)
    plt.ylabel('Number of Mentions', labelpad=15)
    plt.xticks(rotation=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/sentiment/tt/distribution/dist_{topic_id}.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()

# yt
topic_mapping = dict(zip(yt_comments['topic_id'], yt_comments['topic_name']))
sentiment_distribution = yt_comments.groupby(['topic_id', 'Sentiment_category']).size().unstack(fill_value=0)

for topic_id in sentiment_distribution.index:
    topic_name = topic_mapping[topic_id] 
    sentiment_counts = sentiment_distribution.loc[topic_id]
    max_sentiment = sentiment_counts.idxmax()
    colors = ['#8198B6' if sentiment == max_sentiment else '#DDE0E4' for sentiment in sentiment_counts.index]

    plt.figure(figsize=(10, 6))
    sentiment_distribution.loc[topic_id].plot(kind='bar', color=colors, alpha=0.7)
    plt.title(f'Sentiment Distribution for Topic {topic_id}: {topic_name}', pad=20)
    plt.xlabel('Sentiment Category', labelpad=15)
    plt.ylabel('Number of Mentions', labelpad=15)
    plt.xticks(rotation=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/sentiment/yt/distribution/dist_{topic_id}.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()



