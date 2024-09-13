import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

post = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Cleaning/Cleaned Datasets/rhode_post.csv')


# MINOR DATA CLEANING ------------------------------------------
# converting data to date variable
post['timestamp'] = pd.to_datetime(post['timestamp']) # convert

# BASIC STATISTICS ANALYSIS ------------------------------------------
# SUMMARY STATISTICS
# descriptive stats for Likes
likes_desc = post['likesCount'].describe()
print(likes_desc)

# descriptive stats for comment
comment_desc = post['commentsCount'].describe()
print(comment_desc)

# descriptive stats for playCount
playCount_desc = post['videoPlayCount'].describe()
print(playCount_desc)

# descriptive stats for video duration
videoDuration_desc = post['videoDuration'].describe()
print(videoDuration_desc)

# descriptive stats for video count
videoView_desc = post['videoViewCount'].describe()
print(videoView_desc)

# CATEGORICAL STATISTICS
# mentions
post['mentions'] = post['mentions0'].fillna('') + ',' + post['mentions1'].fillna('') + ',' + post['mentions2'].fillna('') + ',' + post['mentions3'].fillna('') + ',' + post['mentions4'].fillna('') + ',' + post['mentions5'].fillna('') + ',' + post['mentions6'].fillna('')
post['mentions'] = post['mentions'].str.split(',').apply(lambda x: [i for i in x if i])

all_mentions = post['mentions'].explode()
unique_users = all_mentions.unique()

print(f"Number of unique users: {len(unique_users)}")
print("Unique users:", unique_users)

# video type
format_counts = post['type'].value_counts()
print(format_counts)

print(post)

# TRENDS OVER TIME ------------------------------------------
# VIDEO UPLOADS OVER TIME
post.set_index('timestamp', inplace=True)
upload_counts = post.resample('M').size()

# The line plot
plt.figure(figsize=(10, 6))
plt.plot(upload_counts.index, upload_counts.values, linestyle='-', color='#8198B6')
plt.title('Number of Video Uploads Over Time', pad=15)
plt.xlabel('Month', labelpad=15)
plt.ylabel('Number of Uploads', labelpad=15)
plt.tight_layout()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/EDA/Figures/posts/video_overtime.png', format='png', bbox_inches='tight', dpi=1200)

# SAVING THE DATASET ------------------------------------------
post = post.drop(columns=['mentions', 'Unnamed: 0'])
post.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/EDA/eda_data/post_ig.csv')
