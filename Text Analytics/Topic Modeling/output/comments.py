import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# READING DATA
ig_comments = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/instagram/data/igcomment_topic.csv")
tt_comments = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/tiktok/data/ttcomment_topic.csv")
yt_comments = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/youtube/data/ytcomment_topic.csv")


# TOPIC DISTRIBUTION EDA ----------------------------
# ig
topic_distribution = ig_comments['topic_id'].value_counts()
topic_distribution = topic_distribution.sort_index()

plt.figure(figsize=(12, 8))
colors = ['#B4C9E2' if value > 400 else '#DDE0E4' for value in topic_distribution]
topic_distribution.plot(kind='bar', color=colors, width=0.7)

plt.title('Topic Distribution for Instagram Comments', pad=20)
plt.xlabel('Topic ID', labelpad=15)
plt.ylabel('Number of Comments', labelpad=15)
plt.xticks(rotation=0, ha='right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/distribution/igc_dist.png', format='png', bbox_inches='tight', dpi=1200)

# tt
topic_distribution = tt_comments['topic_id'].value_counts()
topic_distribution = topic_distribution.sort_index()

plt.figure(figsize=(12, 8))
colors = ['#B4C9E2' if value > 1000 else '#DDE0E4' for value in topic_distribution]
topic_distribution.plot(kind='bar', color=colors, width=0.7)

plt.title('Topic Distribution for TikTok Comments', pad=20)
plt.xlabel('Topic ID', labelpad=15)
plt.ylabel('Number of Comments', labelpad=15)
plt.xticks(rotation=0, ha='right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/distribution/ttc_dist.png', format='png', bbox_inches='tight', dpi=1200)

# yt
topic_distribution = yt_comments['topic_id'].value_counts()
topic_distribution = topic_distribution.sort_index()

plt.figure(figsize=(12, 8))
colors = ['#B4C9E2' if value > 400 else '#DDE0E4' for value in topic_distribution]
topic_distribution.plot(kind='bar', color=colors, width=0.7)

plt.title('Topic Distribution for YouTube Comments', pad=20)
plt.xlabel('Topic ID', labelpad=15)
plt.ylabel('Number of Comments', labelpad=15)
plt.xticks(rotation=0, ha='right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/distribution/ytc_dist.png', format='png', bbox_inches='tight', dpi=1200)

# TOPIC WORD CLOUD ----------------------------
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size < 50:
        return '#DDE0E4'  
    elif font_size > 90:
        return '#8198B6'
    else:
        return '#B4C9E2' 

# ig
for topic_id in ig_comments['topic_id'].unique():
    topic_comments = ig_comments[ig_comments['topic_id'] == topic_id]['Comment']
    text = ' '.join(comment for comment in topic_comments)

    wordcloud = WordCloud(width=1600, height=800, 
                          background_color='white',
                          max_words=200,
                          color_func=custom_color_func).generate(text)

    topic_name = ig_comments[ig_comments['topic_id'] == topic_id]['topic_name'].iloc[0]
    print(f"Topic ID: {topic_id}, Topic Name: {topic_name}")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {topic_name}", pad=20)
    
    plt.savefig(f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/word_cloud/ig/comment/wordcloud_{topic_id}.png", format='png', dpi=1200)
    plt.close()

# tt
for topic_id in tt_comments['topic_id'].unique():
    topic_comments = tt_comments[tt_comments['topic_id'] == topic_id]['text']
    text = ' '.join(comment for comment in topic_comments)

    wordcloud = WordCloud(width=1600, height=800, 
                        background_color='white',
                        max_words=200,
                        color_func=custom_color_func).generate(text)

    topic_name = tt_comments[tt_comments['topic_id'] == topic_id]['topic_name'].iloc[0]
    print(f"Topic ID: {topic_id}, Topic Name: {topic_name}")

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {topic_name}", pad=20)

    plt.savefig(f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/word_cloud/tt/comment/wordcloud_{topic_id}.png", format='png', dpi=1200)
    plt.close()

# yt
for topic_id in yt_comments['topic_id'].unique():
    topic_comments = yt_comments[yt_comments['topic_id'] == topic_id]['Comment']
    text = ' '.join(comment for comment in topic_comments)

    wordcloud = WordCloud(width=1600, height=800, 
                        background_color='white',
                        max_words=200,
                        color_func=custom_color_func).generate(text)

    topic_name = yt_comments[yt_comments['topic_id'] == topic_id]['topic_name'].iloc[0]
    print(f"Topic ID: {topic_id}, Topic Name: {topic_name}")

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {topic_name}", pad=20)

    plt.savefig(f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/word_cloud/yt/comment/wordcloud_{topic_id}.png", format='png', dpi=1200)
    plt.close()