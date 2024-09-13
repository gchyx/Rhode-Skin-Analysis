import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words

import ssl

# nltk.download()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#read data
pd.set_option('display.max_columns', None)

tt_post = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/Data_Extraction/rhodetiktokpost.csv')

# keeping required columns only
required_columns = ['collectCount', 'commentCount', 'createTimeISO', 'diggCount', 'mentions/0', 
                     'mentions/1', 'playCount', 'shareCount', 'text', 'videoMeta/duration', 'videoMeta/format']

# Filter the DataFrame to only include the required columns
tt_post = tt_post[required_columns]

# rename the columns
tt_post.rename(columns={'mentions/0': 'mentions0'}, inplace=True)
tt_post.rename(columns={'mentions/1': 'mentions1'}, inplace=True)
tt_post.rename(columns={'diggCount': 'likes'}, inplace=True)

print(tt_post)

# TEXT NORMALIZATION
tt_post['text'] = tt_post['text'].str.lower()

# removing emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U0001F900-\U0001F9FF"  
                           u"\U00010000-\U0010ffff"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

tt_post['text'] = tt_post['text'].apply(remove_emoji)
print("Cleaning emoji completed...")

tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"-", " ", x))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

tt_post['text'] = tt_post['text'].apply(remove_stopwords)
print("Removing stopwords completed...")

# other cleaning stuff
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"[+:,.]", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"— ", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"• ", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"@ ", "@", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r" ’", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"“ ", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r" ”", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"@\w+", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"! ", "", x))
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"\? ", "", x) if isinstance(x, str) else x)
tt_post['text'] = tt_post['text'].apply(lambda x: re.sub(r"  ", " ", x))


# drop NAs for text
tt_post.dropna(subset=['text'], inplace=True)
# handling missing data
tt_post['text'] = tt_post['text'].str.strip()
tt_post['text'] = tt_post['text'].replace(r'^\s*$', np.nan, regex=True)
tt_post['text'].replace('', pd.NA, inplace=True)
tt_post['text'].replace('nan', pd.NA, inplace=True)
tt_post['text'].replace(np.nan, pd.NA, inplace=True)
tt_post.dropna(subset=['text'], inplace=True)

tt_post.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/Cleaning/Cleaned Datasets/rhode_tiktokpost_cleaned.csv')
