import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

# reading the data
igcomments = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Cleaning/Cleaned Datasets/cleaned_rhode.csv')
print(igcomments)

# setting up the data
# converting data to date variable
igcomments['timestamp'] = pd.to_datetime(igcomments['timestamp']) # convert
igcomments['timestamp'] = igcomments['timestamp'].dt.date # extract date only

# BASIC DESCRIPTIVE ANALYSIS ------------------------------------------
# descriptive stats for Likes
likes_desc = igcomments['likesCount'].describe()
print(likes_desc)

# COUNT AND FREQUENCY
# unique username
unique_users_count = igcomments['ownerUsername'].nunique()
print(f"Number of unique ownerUsernames: {unique_users_count}")

# unique videoID
unique_post_count = igcomments['postUrl'].nunique()
post_freq = igcomments['postUrl'].value_counts()
print(f"Number of unique posts: {unique_post_count}")
print(post_freq, "\n")

# total number of comments
total_comments = igcomments['Comment'].count()
print(f"Total Number of Comments: {total_comments}")


# USER ANALYSIS ------------------------------------------
total_likes_by_user = igcomments.groupby('ownerUsername')['likesCount'].sum().sort_values(ascending=False)
top_n = 5 
top_users = total_likes_by_user.head(top_n)

# The bar plot
plt.figure(figsize=(12, 8))
top_users.plot(kind='barh', color='#B4C9E2', edgecolor='#8198B6')
plt.title('Top Users by Total Likes Received', pad=20)
plt.xlabel('Total Likes', labelpad=15)
plt.ylabel('User', labelpad=15)
plt.gca().invert_yaxis()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()

plt.savefig('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/EDA/Figures/comments/igcomments_topusers.png', format='png', bbox_inches='tight', dpi=1200)


# Word Cloud ------------------------------------------
text = ' '.join(comment for comment in igcomments['Comment'])

def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if font_size < 50:
        return '#DDE0E4'  
    elif font_size > 90:
        return '#8198B6'
    else:
        return '#B4C9E2' 

# The wordcloud
stopwords = set(['hailey', 'actually', 'even', 'think', 'much', 'know', 'please', 'pls'
                 'thats', 'thing', 'get', 'make', 'would', 'say', 'see', 'look', 'product','products'])

wordcloud = WordCloud(width=1600, height=800, 
                      background_color='white',
                      stopwords=stopwords,  
                      max_words=200,
                      color_func=custom_color_func).generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/EDA/Figures/comments/igcomments_wordcloud.png', format='png', bbox_inches='tight', dpi=1200) 

igcomments.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/EDA/eda_data/comments_ig.csv')