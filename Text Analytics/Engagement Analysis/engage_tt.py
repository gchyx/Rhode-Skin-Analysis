import pandas as pd
from datetime import datetime

engage_tt = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/Cleaning/Cleaned Datasets/rhode_tiktokpost_cleaned.csv')
print(engage_tt.head())

# creating new dataset to add engagement rates monthly
engage_tt['engagement_rate'] = ((engage_tt['likes'] + engage_tt['commentCount']) / engage_tt['likes'])
avg_engagement_rate = engage_tt['engagement_rate'].mean()
engage_tt['createTimeISO'] = pd.to_datetime(engage_tt['createTimeISO'])

# monthly engagement rate
monthly_data = engage_tt.groupby(pd.Grouper(key='createTimeISO', freq='ME')).agg(
    total_likes=('likes', 'sum'),
    total_comments=('commentCount', 'sum'),
    total_collects=('collectCount', 'sum'),
    total_shares=('shareCount', 'sum'),
    total_posts=('text', 'count')
)

monthly_data['avg_engagement_rate'] = (
    (monthly_data['total_likes'] + monthly_data['total_comments'] + monthly_data['total_collects'] + monthly_data['total_shares']) / monthly_data['total_posts']) / 941500 * 100
monthly_data['avg_engagement_rate'] = monthly_data['avg_engagement_rate'].round(2)
print(monthly_data)

monthly_data.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Engagement Analysis/data/tt/engagement_month.csv', index=True)

# day of the week engagement rate
engage_tt['day_of_week'] = engage_tt['createTimeISO'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
engage_tt['day_of_week'] = pd.Categorical(engage_tt['day_of_week'], categories=day_order, ordered=True)

daily_data = engage_tt.groupby('day_of_week').agg(
    total_likes=('likes', 'sum'),
    total_comments=('commentCount', 'sum'),
    total_collects=('collectCount', 'sum'),
    total_shares=('shareCount', 'sum'),
    total_posts=('text', 'count')
)

daily_data['avg_engagement_rate'] = (
    (daily_data['total_likes'] + daily_data['total_comments'] + daily_data['total_collects'] + daily_data['total_shares']) / daily_data['total_posts']) / 941500 * 100
daily_data['avg_engagement_rate'] = daily_data['avg_engagement_rate'].round(2)
daily_data = daily_data.sort_index()
print(daily_data)

daily_data.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Engagement Analysis/data/tt/engagement_day.csv', index=True)
