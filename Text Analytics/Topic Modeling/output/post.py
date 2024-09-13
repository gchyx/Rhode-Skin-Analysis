import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# READING DATA
ig_post = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/instagram/data/igpost_topic.csv")
tt_post = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/tiktok/data/ttpost_topic.csv")


# TOPIC DISTRIBUTION EDA ----------------------------
# ig
topic_distribution = ig_post['topic_id'].value_counts()
topic_distribution = topic_distribution.sort_index()

plt.figure(figsize=(12, 8))
colors = ['#B4C9E2' if value > 20 else '#DDE0E4' for value in topic_distribution]
topic_distribution.plot(kind='bar', color=colors, width=0.7)

plt.title('Topic Distribution for Instagram Post', pad=20)
plt.xlabel('Topic ID', labelpad=15)
plt.ylabel('Number of Comments', labelpad=15)
plt.xticks(rotation=0, ha='right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/distribution/igp_dist.png', format='png', bbox_inches='tight', dpi=1200)

# tt
topic_distribution = tt_post['topic_id'].value_counts()
topic_distribution = topic_distribution.sort_index()

plt.figure(figsize=(12, 8))
colors = ['#B4C9E2' if value > 50 else '#DDE0E4' for value in topic_distribution]
topic_distribution.plot(kind='bar', color=colors, width=0.7)

plt.title('Topic Distribution for TikTok Post', pad=20)
plt.xlabel('Topic ID', labelpad=15)
plt.ylabel('Number of Comments', labelpad=15)
plt.xticks(rotation=0, ha='right')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/distribution/ttp_dist.png', format='png', bbox_inches='tight', dpi=1200)

# TOPIC WORD CLOUD ----------------------------
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size < 50:
        return '#DDE0E4'  
    elif font_size > 90:
        return '#8198B6'
    else:
        return '#B4C9E2' 

# ig
for topic_id in ig_post['topic_id'].unique():
    topic_comments = ig_post[ig_post['topic_id'] == topic_id]['caption']
    text = ' '.join(comment for comment in topic_comments)

    wordcloud = WordCloud(width=1600, height=800, 
                          background_color='white',
                          max_words=200,
                          color_func=custom_color_func).generate(text)

    topic_name = ig_post[ig_post['topic_id'] == topic_id]['topic_name'].iloc[0]
    print(f"Topic ID: {topic_id}, Topic Name: {topic_name}")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {topic_name}", pad=20)
    
    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/word_cloud/ig/post/wordcloud_{topic_id}.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()

# tt
for topic_id in tt_post['topic_id'].unique():
    topic_comments = tt_post[tt_post['topic_id'] == topic_id]['text']
    text = ' '.join(comment for comment in topic_comments)

    wordcloud = WordCloud(width=1600, height=800, 
                        background_color='white',
                        max_words=200,
                        color_func=custom_color_func).generate(text)

    topic_name = tt_post[tt_post['topic_id'] == topic_id]['topic_name'].iloc[0]
    print(f"Topic ID: {topic_id}, Topic Name: {topic_name}")

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {topic_name}", pad=20)

    filename = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/word_cloud/tt/post/wordcloud_{topic_id}.png"
    plt.savefig(filename, format='png', dpi=1200)
    plt.close()


# TOPIC OVER TIME ----------------------------
# ig
topic_mapping = dict(zip(ig_post['topic_id'], ig_post['topic_name']))
ig_post['timestamp'] = pd.to_datetime(ig_post['timestamp'])
topic_evolution = ig_post.groupby([pd.Grouper(key='timestamp', freq='W'), 'topic_id']).size().unstack().fillna(0)

for topic in topic_evolution.columns:
    topic_name = topic_mapping[topic]
    plt.figure(figsize=(10, 6))
    plt.plot(topic_evolution.index, topic_evolution[topic], color='#8198B6', linewidth=2)
    plt.title(f'{topic_name} Evolution Over Time', pad=20)
    plt.xlabel('Time', labelpad=15)
    plt.ylabel('Number of Mentions', labelpad=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    file_name = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/overtime/ig/topic_{topic}_overtime.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=1200)
    plt.close()

# tt
topic_mapping = dict(zip(tt_post['topic_id'], tt_post['topic_name']))
tt_post['createTimeISO'] = pd.to_datetime(tt_post['createTimeISO'])
topic_evolution = tt_post.groupby([pd.Grouper(key='createTimeISO', freq='W'), 'topic_id']).size().unstack().fillna(0)

for topic in topic_evolution.columns:
    topic_name = topic_mapping[topic]
    plt.figure(figsize=(10, 6))
    plt.plot(topic_evolution.index, topic_evolution[topic], color='#8198B6', linewidth=2)
    plt.title(f'{topic_name} Evolution Over Time', pad=20)
    plt.xlabel('Time', labelpad=15)
    plt.ylabel('Number of Mentions', labelpad=15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    file_name = f"/Users/gladys/Documents/GitHub/SMA_A02/Text Analytics/Topic Modeling/output/figures/overtime/tt/topic_{topic}_overtime.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=1200)
    plt.close()