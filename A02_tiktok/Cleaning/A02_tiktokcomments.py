import pandas as pd
import numpy as np
import re
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.corpus import words

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

'''
nltk.download()
'''

pd.set_option('display.max_columns', None)

# reading data
rhode_tiktok = pd.read_csv("/Users/gladys/Documents/GitHub/SMA_A02/A02_tiktok/Data_Extraction/rhodetiktokcomments.csv")


# delete columns
rhode_tiktok = rhode_tiktok.drop(['cid', 'createTime', 'submittedVideoUrl', 'uid','repliesToId','avatarThumbnail', 'replyCommentTotal'], axis=1)

rhode_tiktok.rename(columns={'diggCount': 'Likes'}, inplace=True)

# removing missing data
rhode_tiktok.dropna(subset=['text'], inplace=True)
rhode_tiktok.dropna()

# TEXT NORMALIZATION
# changing to lowercase
rhode_tiktok['text'] = rhode_tiktok['text'].str.lower()

# removing emoji
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # other symbols
        u"\U000024C2-\U0001F251"  # enclosed characters
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

rhode_tiktok['text'] = rhode_tiktok['text'].apply(remove_emojis)
print("Emoji cleaning completed...")

# removing non english comments
def remove_non_english(text):
    # This pattern matches anything that is not an English letter, digit, or common punctuation
    pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'"-]+')
    return pattern.sub('', text)

rhode_tiktok['text'] = rhode_tiktok['text'].apply(remove_non_english)

# removing non english language comments
def filter_nonenglish_comments(comment):
    try:
        lang = detect(comment)
        if lang in ['es', 'pt', 'ms', 'fr', 'it', 'de']:
            return False  # filter out comments in these languages
        else:
            return True  # keep comments in other languages
    except:
        return True  # if language detection fails 

rhode_tiktok = rhode_tiktok[rhode_tiktok['text'].apply(filter_nonenglish_comments)]
print("Removing non-english comments completed...")

# Cleaning out spam and bot patterns
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
# Bot patterns
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

# Compile the combined pattern
combined_patterns = spam_patterns + bot_patterns
pattern = re.compile('|'.join(map(re.escape, combined_patterns)), flags=re.IGNORECASE)

# Filter out rows that contain any of the spam or bot patterns
def remove_spambots(comment):
    return not pattern.search(comment)

# check spelling
# Correcting spelling (this will take forever and idk when)
nltk.download('words')
word_list = set(words.words())

# Apply the filter to the DataFrame
rhode_tiktok =  rhode_tiktok[rhode_tiktok['text'].apply(remove_spambots)]

# Define the correct_word function to correct exaggerated words
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
    corrected_comment = correct_text(row['text'])
    return index, corrected_comment


# Process each row sequentially
for index, row in rhode_tiktok.iterrows():
    index, corrected_comment = process_row(index, row)
    rhode_tiktok.at[index, 'text'] = corrected_comment
    print(f"Row {index + 1} done.")

print("Correcting spelling mistakes FINALLY completed...")


# removing some stuff
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub('|'.join(map(re.escape, spam_patterns)), '', x, flags=re.IGNORECASE))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub('|'.join(map(re.escape, bot_patterns)), '', x, flags=re.IGNORECASE))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"'re", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"``", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"\d+", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'[^\w\s]', "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'\b\w\b', " ", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"  ", " ", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"cant", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"wont", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"yall", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"dont", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"im ", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"shes", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"bout", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r"youll", "", x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'\bhello\w*\b', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'\bhey\w*\b', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'hi ', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'hii', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'hiii', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'hiiii', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'hiiiii', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'justin', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r' tho ', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'okay', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'omg', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'lol', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'lmao', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'oh ', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'http\S+', '', x))
rhode_tiktok['text'] = rhode_tiktok['text'].apply(lambda x: re.sub(r'<.*?>', '', x))

# removing stop-words
nltk.download('stopwords')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

rhode_tiktok['text'] = rhode_tiktok['text'].apply(remove_stopwords)
print("Removing stopwords completed...")

# handling missing data
rhode_tiktok['text'] = rhode_tiktok['text'].str.strip()
rhode_tiktok['text'] = rhode_tiktok['text'].replace(r'^\s*$', np.nan, regex=True)
rhode_tiktok.dropna(subset=['text'], inplace=True)

# deleting duplicates
rhode_tiktok['text'] = rhode_tiktok['text'].drop_duplicates()

print(rhode_tiktok)

rhode_tiktok.dropna(subset=['text'], inplace=True)

rhode_tiktok.to_csv('/Users/gladys/Documents/GitHub/SMA_A02/A02_tiktok/Cleaning/Cleaned Datasets/rhode_tiktokcomments_cleaned.csv')
