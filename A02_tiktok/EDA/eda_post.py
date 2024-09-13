import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

post = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/Cleaning/Cleaned Datasets/rhode_tiktokpost_cleaned.csv')


# MINOR DATA CLEANING ------------------------------------------
# converting data to date variable
post['createTimeISO'] = pd.to_datetime(post['createTimeISO']) # convert

# Rename the column
post.rename(columns={'videoMeta/duration': 'videoDuration'}, inplace=True)
post.rename(columns={'videoMeta/format': 'videoFormat'}, inplace=True)

# if video format is NA, change to 'post'
post['videoFormat'] = post['videoFormat'].fillna("jpeg")
print(post)

# if video duration is 0, change to 'NA'
post['videoDuration'] = post['videoDuration'].replace(0, np.nan)

# BASIC STATISTICS ANALYSIS ------------------------------------------
# SUMMARY STATISTICS
# descriptive stats for Likes
likes_desc = post['likes'].describe()
print(likes_desc)

# descriptive stats for collect
collect_desc = post['collectCount'].describe()
print(collect_desc)

# descriptive stats for comment
comment_desc = post['commentCount'].describe()
print(comment_desc)

# descriptive stats for share
share_desc = post['shareCount'].describe()
print(share_desc)

# descriptive stats for playCount
playCount_desc = post['playCount'].describe()
print(playCount_desc)

# descriptive stats for video duration
videoDuration_desc = post['videoDuration'].describe()
print(videoDuration_desc)

# CATEGORICAL STATISTICS
# mentions
post['mentions'] = post['mentions0'].fillna('') + ',' + post['mentions1'].fillna('')
post['mentions'] = post['mentions'].str.split(',').apply(lambda x: [i for i in x if i])

all_mentions = post['mentions'].explode()
unique_users = all_mentions.unique()

print(f"Number of unique users: {len(unique_users)}")
print("Unique users:", unique_users)

# video format
format_counts = post['videoFormat'].value_counts()
print(format_counts)

print(post)

# TRENDS OVER TIME ------------------------------------------
# VIDEO UPLOADS OVER TIME
post.set_index('createTimeISO', inplace=True)
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

plt.savefig('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/EDA/Figures/posts/video_overtime.png', format='png', bbox_inches='tight', dpi=1200)

# SAVING THE DATASET ------------------------------------------
post = post.drop(columns=['mentions', 'Unnamed: 0'])
post.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/EDA/eda_data/post_tt.csv')
