import pandas as pd
from datetime import datetime

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

engage_data = pd.read_csv("C:/Users/kobee/OneDrive/Desktop/rhode_tiktokposts_cleaned.csv")

df = engage_data

print(df.head())

df['engagement_rate'] = ((df['likes'] + df['commentCount']) / df['likes'])

print(df['engagement_rate'])

avg_engagement_rate = df['engagement_rate'].mean()
print(f"The average engagement rate is: {avg_engagement_rate}")

df['createTimeISO'] = pd.to_datetime(df['createTimeISO'])

# MONTH
monthly_data = df.groupby(pd.Grouper(key='createTimeISO', freq='ME')).agg(
    total_likes=('likes', 'sum'),
    total_comments=('commentCount', 'sum'),
    total_posts=('PostID', 'count'),
)
monthly_data['avg_engagement_rate'] = ((monthly_data['total_likes'] + monthly_data['total_comments']) / monthly_data['total_posts'] / 941500) * 100
monthly_data['avg_engagement_rate'] = monthly_data['avg_engagement_rate'].round(2)
print(monthly_data)

# DAY
df['day_of_week'] = df['createTimeISO'].dt.dayofweek

day_of_week_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df['day_of_week'] = df['day_of_week'].map(dict(enumerate(day_of_week_labels)))

avg_engagement_by_day_of_week = df.groupby('day_of_week')['engagement_rate'].mean().round(2)

print("\nAverage engagement rate by day of week:")

print(avg_engagement_by_day_of_week)


# TIME
df['hour_of_day'] = df['createTimeISO'].dt.hour
df['engagement_rate'] = df['likes'] + df['commentCount']

avg_engagement_by_hour = df.groupby('hour_of_day')['engagement_rate'].mean().round(2)

time_of_day_bins = [0, 6, 12, 18, 24]
time_of_day_labels = ['Night', 'Morning', 'Afternoon', 'Evening']

df['time_of_day'] = pd.cut(df['hour_of_day'], bins=time_of_day_bins, labels=time_of_day_labels)

avg_engagement_by_time_of_day = df.groupby('time_of_day')['engagement_rate'].mean().round(2)

print("\nAverage engagement rate by time of day:")

print(avg_engagement_by_time_of_day)

average_comments = monthly_data['total_comments'].mean()

print(f"Average comments count: {round(average_comments)}")

print("Average engagement rate: {:.2f}%".format(avg_engagement_rate * 100))

df['engagement_type'] = df.apply(lambda row: 'High' if row['engagement_rate'] > 0.1 else ('Medium' if row['engagement_rate'] > 0.05 else 'Low'), axis=1)

avg_engagement_rate = df['engagement_rate'].mean()

top_10_posts = df.nlargest(10, 'engagement_rate')
bottom_10_posts = df.nsmallest(10, 'engagement_rate')

print("Top 10 most engaging posts:")
print(top_10_posts)
print("Bottom 10 least engaging posts:")
print(bottom_10_posts)


monthly_data.to_csv('tiktok_engaged_monthly_data.csv')

avg_engagement_by_day_of_week.to_csv('tiktok_engaged_day.csv')

avg_engagement_by_time_of_day.to_csv('tiktok_engaged_time.csv')

top_10_posts.to_csv('tiktok_engaged_top10.csv')

bottom_10_posts.to_csv('tiktok_engaged_bottom10.csv')