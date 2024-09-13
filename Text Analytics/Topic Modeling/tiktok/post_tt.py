import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim

# download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# reading the data
post = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/EDA/eda_data/post_tt.csv')

print(post)

# DATA PREPROCESSING
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Tokenize, remove stop words, and lemmatize
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

post['text'] = post['text'].apply(preprocess)

# CREATE DICTIONARY
dictionary = gensim.corpora.Dictionary(post['text'])
corpus = [dictionary.doc2bow(text) for text in post['text']]
dictionary.save('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/gensim/dictionary_post.gensim')

'''
# FINDING OPTIMAL NUMBER OF TOPICS
def compute_coherence_values(dictionary, corpus, texts, num_topics_range):
    coherence_values = []
    for num_topics in num_topics_range:
        try:
            model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics,
                                           random_state=100,
                                           passes=10,
                                           alpha='auto')
            coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())
        except Exception as e:
            print(f"Error with num_topics={num_topics}: {e}")
            coherence_values.append(None)  # Append None if there's an error
    return coherence_values

# range for number of topics
num_topics_range = [5, 10, 15, 20, 40]

if __name__ == '__main__':
    try:
        # Assuming 'post['caption']' is the list of processed texts
        coherence_values = compute_coherence_values(dictionary=dictionary, 
                                                    corpus=corpus, 
                                                    texts=post['text'], 
                                                    num_topics_range=num_topics_range)
        
        # filter out None values (in case of errors)
        valid_indices = [i for i, val in enumerate(coherence_values) if val is not None]
        valid_num_topics = [num_topics_range[i] for i in valid_indices]
        valid_coherence_values = [coherence_values[i] for i in valid_indices]

        # plot coherence scores
        plt.figure()
        plt.plot(valid_num_topics, valid_coherence_values, marker='o', linestyle='-')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Coherence Score vs Number of Topics')
        plt.show()

        # find the optimal number of topics
        optimal_num_topics = valid_num_topics[valid_coherence_values.index(max(valid_coherence_values))]
        print(f"Optimal number of topics: {optimal_num_topics}")

        # plot is saved manually from the plt.show() to avoid running this hell code again.
        # results is around 20 topics

    except Exception as e:
        print(f"An error occurred: {e}")
'''

# LDA MODEL
num_topics = 17
random_seed = 70
lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=random_seed)

topics = lda_model.print_topics(num_words=15)
for topic in topics:
    print(topic)


# putting results into dataset
topic_list = []
for topic in topics:
    topic_id, topic_words = topic
    words = [word.split('*')[1].strip('"') for word in topic_words.split(' + ')]
    # Create a dictionary for the topic
    topic_data = {'Topic ID': topic_id}
    for i, word in enumerate(words):
        topic_data[f'Word {i+1}'] = word
    topic_list.append(topic_data)

topics_df = pd.DataFrame(topic_list)
topics_df.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/data/topics_post.csv', index=False)
lda_model.save('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/gensim/lda_model_post.gensim')

# VISUALIZE
lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_display, '/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/figures/lda_visualization.html')


# PUTTING THE TOPIC ACCORDING TO THE CAPTIONS IN THE DATASET
lda_model = LdaModel.load('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/gensim/lda_model_post.gensim')
dictionary = Dictionary.load('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/gensim/dictionary_post.gensim')

# Load your new dataset
post_tm = pd.read_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/A02_tiktok/EDA/eda_data/post_tt.csv')


# convert new captions (already preprocessed) to BOW format
new_corpus = [dictionary.doc2bow(text.split()) for text in post_tm['text']]

# infer topics
def infer_topics(corpus, model):
    return [model.get_document_topics(doc) for doc in corpus]

new_topic_distributions = infer_topics(new_corpus, lda_model)

# assign the topic to each caption
post_tm['topic_id'] = [max(doc, key=lambda x: x[1])[0] if doc else None for doc in new_topic_distributions]

# naming topics
topic_names = {
    0: "Day & Night Rhode Skincare Routine",
    1: "Rhode Lip Products & Shades",
    2: "Rhode Pop-Up Event & New Products",
    3: "Rhode Kit & Event",
    4: "Rhode Pocket Blush Shades in NYC",
    5: "Rhode Summer Lip Tints and Pocket Blush",
    6: "Lip Tint Pop-Up",
    7: "Pocket Blush Shades",
    8: "Pineapple Refresh Cleanser",
    9: "Products for Summer Makeup Look",
    10: "Rhode Glazing Milk & Reviews",
    11: "Skincare & Makeup Routine Recommendations",
    12: "Pineapple Refresh Cleanser Formula",
    13: "Lip Tint & Case Waitlist",
    14: "Lip & Skincare Products",
    15: "Rhode Products & International Shipping",
    16: "Lip Tint Application & Formula"
}

def get_topic_name(topic_id):
    return topic_names.get(topic_id, "Unknown Topic")

post_tm['topic_name'] = post_tm['topic_id'].apply(get_topic_name)

post_tm.to_csv('/Users/gladys/Documents/GitHub/Rhode-Skin-Analysis/Text Analytics/Topic Modeling/tiktok/data/ttpost_topic.csv', index=False)
print(post_tm.head())
