import pandas as pd
import numpy as np
import re
import nltk
from langdetect import detect
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

rhode = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Data_extraction/rhodeigcomment.csv')
rhode.rename(columns={'text': 'Comment'}, inplace=True)
rhode.dropna(subset=['Comment'], inplace=True)

# TEXT NORMALIZATION
rhode['Comment'] = rhode['Comment'].str.lower()

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

rhode['Comment'] = rhode['Comment'].apply(remove_emoji)
print("Emoji Cleaned")


def remove_non_english(comment):
    pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'"-]+')
    return pattern.sub('', comment)

rhode['Comment'] = rhode['Comment'].apply(remove_non_english)

def filter_nonenglish_comments(comment):
    try:
        lang = detect(comment)
        if lang in ['es', 'pt', 'ms', 'fr', 'it', 'de']:
            return False  # filter out comments in these languages
        else:
            return True  # keep comments in other languages
    except:
        return True  # if language detection fails 

rhode = rhode[rhode['Comment'].apply(filter_nonenglish_comments)]
print("Non-English Comments Removed")

spam_patterns = [
    "check out my channel",
    "free giveaway",
    "get rich quick",
    "click the link",
    "subscribe to my channel",
    "earn money online",
    "make $1000 a day",
    "lose weight fast",
    "buy followers",
    "watch my video",
    "earn",
    "commission",
    "affiliate"
]

bot_patterns = [
    "I'm a bot",
    "automated message",
    "subscribe to my channel",
    "great video, check out mine",
    "earn money online",
    "free followers",
    "get rich quick",
    "comment below for a shoutout",
    "follow for follow",
    "click the link in my bio"
]

combined_patterns = spam_patterns + bot_patterns
pattern = re.compile('|'.join(map(re.escape, combined_patterns)), flags=re.IGNORECASE)

def remove_spambots(comment):
    return not pattern.search(comment)

# Apply the filter to the DataFrame
rhode = rhode[rhode['Comment'].apply(remove_spambots)]

def correct_word(word):
    corrected_word = re.sub(r'(.)\1{2,}', r'\1', word)
    return corrected_word

# Define the correct_text function
def correct_text(text):
    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    return ' '.join(corrected_words)

# Define a function to process a single row
def process_row(index, row):
    corrected_comment = correct_text(row['Comment'])
    return index, corrected_comment


# Process each row sequentially
for index, row in rhode.iterrows():
    index, corrected_comment = process_row(index, row)
    rhode.at[index, 'Comment'] = corrected_comment
    print(f"Row {index + 1} done.")

print("Spelling mistakes corrected")

rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub('|'.join(map(re.escape, spam_patterns)), '', x, flags=re.IGNORECASE))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub('|'.join(map(re.escape, bot_patterns)), '', x, flags=re.IGNORECASE))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"'re", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"``", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"\d+", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'[^\w\s]', "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'\b\w\b', " ", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"  ", " ", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"r6alr8", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"skyoceanmountainsnow", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"themakenzielynn", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'fid rveievr', "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"flexgurl ", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"hm", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"bout", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r"youll", "", x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'\bhello\w*\b', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'\bhey\w*\b', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'hi ', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'hii', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'katevasq', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'aamiin', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'omd', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'vem pro br', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r' tho ', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'okay', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'lol', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'lmao', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'oh ', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'http\S+', '', x))
rhode['Comment'] = rhode['Comment'].apply(lambda x: re.sub(r'<.*?>', '', x))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

rhode['Comment'] = rhode['Comment'].apply(remove_stopwords)
print("Removing stopwords completed...")

# handling missing data
rhode['Comment'] = rhode['Comment'].str.strip()
rhode['Comment'] = rhode['Comment'].replace(r'^\s*$', np.nan, regex=True)
rhode.dropna(subset=['Comment'], inplace=True)

# deleting duplicates
rhode['Comment'] = rhode['Comment'].drop_duplicates()

print(rhode)

# drop missing and irrelevant column
rhode.dropna(subset=['Comment'], inplace=True)
rhode = rhode.drop(columns=['ownerProfilePicUrl'])

print(rhode)

rhode.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_IG/Cleaning/Cleaned Datasets/cleaned_rhode.csv')