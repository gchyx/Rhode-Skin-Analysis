import pandas as pd
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

r_post = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Data_extraction/rhodeigpost.csv')

# keeping required columns only
required_columns = ['caption', 'commentsCount', 'likesCount', 'mentions/0', 
                     'mentions/1', 'mentions/2', 'mentions/3', 'mentions/4', 'mentions/5', 
                     'mentions/6', 'timestamp', 'type', 'videoDuration', 'videoPlayCount', 'videoViewCount']


# Filter the DataFrame to only include the required columns
r_post = r_post[required_columns]

# rename the columns
r_post.rename(columns={'mentions/0': 'mentions0'}, inplace=True)
r_post.rename(columns={'mentions/1': 'mentions1'}, inplace=True)
r_post.rename(columns={'mentions/2': 'mentions2'}, inplace=True)
r_post.rename(columns={'mentions/3': 'mentions3'}, inplace=True)
r_post.rename(columns={'mentions/4': 'mentions4'}, inplace=True)
r_post.rename(columns={'mentions/5': 'mentions5'}, inplace=True)
r_post.rename(columns={'mentions/6': 'mentions6'}, inplace=True)

print(r_post)

# TEXT NORMALIZATION
r_post['caption'] = r_post['caption'].str.lower()

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
                           u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
                           u"\U00010000-\U0010ffff"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

r_post['caption'] = r_post['caption'].apply(remove_emoji)
print("Cleaning emoji completed...")

r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"-", " ", x))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

r_post['caption'] = r_post['caption'].apply(remove_stopwords)
print("Removing stopwords completed...")

# other cleaning stuff
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"[+:,.]", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"— ", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"• ", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"@ ", "@", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r" ’", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"“ ", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r" ”", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"@\w+", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"! ", "", x))
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"\? ", "", x) if isinstance(x, str) else x)
r_post['caption'] = r_post['caption'].apply(lambda x: re.sub(r"  ", " ", x))

r_post['mentions0'] = r_post['mentions0'].apply(lambda x: re.sub(r",", "", x) if isinstance(x, str) else x)
r_post['mentions0'] = r_post['mentions0'].apply(lambda x: re.sub(r"’s", "", x) if isinstance(x, str) else x)
r_post['mentions1'] = r_post['mentions1'].apply(lambda x: re.sub(r",", "", x) if isinstance(x, str) else x)
r_post['mentions1'] = r_post['mentions1'].apply(lambda x: re.sub(r"’s", "", x) if isinstance(x, str) else x)
r_post['mentions2'] = r_post['mentions2'].apply(lambda x: re.sub(r"\.", "", x) if isinstance(x, str) else x)

r_post['mentions0'] = r_post['mentions0'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)
r_post['mentions1'] = r_post['mentions1'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)
r_post['mentions2'] = r_post['mentions2'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)
r_post['mentions3'] = r_post['mentions3'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)
r_post['mentions4'] = r_post['mentions4'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)
r_post['mentions5'] = r_post['mentions5'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)
r_post['mentions6'] = r_post['mentions6'].apply(lambda x: ' '.join('@' + word for word in x.split()) if isinstance(x, str) else x)


# drop NAs for caption
r_post.dropna(subset=['caption'], inplace=True)
r_post.drop([90, 139], inplace=True)

r_post.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Cleaning/Cleaned Datasets/rhode_post.csv')
