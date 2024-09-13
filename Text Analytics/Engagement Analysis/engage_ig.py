import pandas as pd
from datetime import datetime

engage_ig = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Cleaning/Cleaned Datasets/rhode_post.csv')
print(engage_ig.head())

# creating new dataset to add engagement rates monthly
engage_ig['engagement_rate'] = ((engage_ig['likesCount'] + engage_ig['commentsCount']) / engage_ig['likesCount'])
avg_engagement_rate = engage_ig['engagement_rate'].mean()
engage_ig['timestamp'] = pd.to_datetime(engage_ig['timestamp'])

# monthly engagement rate
monthly_data = engage_ig.groupby(pd.Grouper(key='timestamp', freq='ME')).agg(
    total_likes=('likesCount', 'sum'),
    total_comments=('commentsCount', 'sum'),
    total_posts=('caption', 'count'),
)

monthly_data['avg_engagement_rate'] = ((monthly_data['total_likes'] + monthly_data['total_comments']) / monthly_data['total_posts'])/2000000*100
monthly_data['avg_engagement_rate'] = monthly_data['avg_engagement_rate'].round(2)
print(monthly_data)

monthly_data.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Engagement Analysis/data/ig/engagement_month.csv', index=True)

# day of the week engagement rate
engage_ig['day_of_week'] = engage_ig['timestamp'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
engage_ig['day_of_week'] = pd.Categorical(engage_ig['day_of_week'], categories=day_order, ordered=True)

daily_data = engage_ig.groupby('day_of_week').agg(
    total_likes=('likesCount', 'sum'),
    total_comments=('commentsCount', 'sum'),
    total_posts=('caption', 'count'),  # Count the number of posts per day
)

daily_data['avg_engagement_rate'] = ((daily_data['total_likes'] + daily_data['total_comments']) / daily_data['total_posts']) / 2000000 * 100
daily_data['avg_engagement_rate'] = daily_data['avg_engagement_rate'].round(2)
daily_data = daily_data.sort_index()
print(daily_data)

daily_data.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Engagement Analysis/data/ig/engagement_day.csv', index=True)
