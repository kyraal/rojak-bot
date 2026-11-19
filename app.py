import streamlit as st
import speech_recognition as sr
import io
import re
import os
import random
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import wordnet
from nltk.wsd import lesk
from gensim.models import Word2Vec

# ==========================================
# 0. INITIALIZATION & NLTK DOWNLOADS
# ==========================================
@st.cache_resource
def download_nltk_data():
    """[SYLLABUS: NLP Toolkits & Preprocessing]"""
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True) # <-- Add this line
    nltk.download('wordnet', quiet=True)
download_nltk_data()

KB_FILE = 'knowledge_base.csv'
LOG_FILE = 'chat_logs.csv'

# ==========================================
# 1. KNOWLEDGE BASE & PERSISTENT SAVING
# ==========================================
def load_kb():
    if os.path.exists(KB_FILE):
        return pd.read_csv(KB_FILE)
    else:
        # Default Knowledge Base
        data = {
            'Intent': ['track_order', 'missing_item', 'damaged_item', 'refund_request'],
            'Sample_Queries': [
                'where is my parcel when will barang arrive order late', 
                'barang tiada missing item did not get parcel',
                'barang rosak damaged broken hancur', 
                'nak balik money refund claim duit'
            ],
            'Response_Variations': [
                'You can track your order at the My Purchases tab.|Check the My Purchases tab to track it.|Order tracking is available in your account.',
                'Sorry about the missing item. Click Dispute to contact the seller.|Please click Dispute for missing items.|Contact the seller via the Dispute button.',
                'If damaged, take a photo and submit a return within 3 days.|For broken items, submit a return request with photos.|Please request a return with picture proof.',
                'Refunds take 3 to 5 working days to reflect.|Your money will be refunded in 3-5 days.|Expect your refund in a few working days.'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(KB_FILE, index=False)
        return df

def save_to_kb(intent, new_query, bot_response):
    df = load_kb()
    new_row = pd.DataFrame({'Intent': [intent], 'Sample_Queries': [new_query], 'Response_Variations': [bot_response]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(KB_FILE, index=False)
    st.session_state.kb_df = df

def log_chat(user_input):
    df = pd.DataFrame({'query': [user_input]})
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

if 'kb_df' not in st.session_state:
    st.session_state.kb_df = load_kb()

# ==========================================
# 2. NLP PIPELINE (SYLLABUS IMPLEMENTATIONS)
# ==========================================

# [SYLLABUS: Word2Vec] Train a mini Word2Vec model on our queries to learn word relationships
sentences = [query.split() for query in st.session_state.kb_df['Sample_Queries']]
w2v_model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=1)

rojak_dict = {"brg": "barang", "x": "tak", "takde": "tiada", "bila": "when", "rosak": "damaged", "duit": "money"}
negative_keywords = ['bodoh', 'babi', 'gila', 'teruk', 'lambat', 'marah']

def clean_rojak_text(text):
    """[SYLLABUS: REGEX, Metacharacters & Morphology]"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'(.)\1+', r'\1\1', text) # Remove repeated chars (e.g., lambaaat -> lambat)
    cleaned_words = [rojak_dict.get(word, word) for word in text.split()]
    return ' '.join(cleaned_words)

def perform_pos_and_wsd(cleaned_text):
    """[SYLLABUS: Syntactic Analysis, POS Tagging & WSD]"""
    tokens = nltk.word_tokenize(cleaned_text)
    pos_tags = nltk.pos_tag(tokens) # POS Tagging
    
    # Word Sense Disambiguation using Lesk algorithm (trying to find noun meaning of 'order' vs verb)
    disambiguated = []
    for token in tokens:
        synset = lesk(tokens, token)
        sense = synset.name() if synset else "unknown"
        disambiguated.append(f"{token}({sense})")
        
    return pos_tags, disambiguated

def analyze_sentiment(cleaned_text):
    """[SYLLABUS: Sentiment Analysis]"""
    if any(word in negative_keywords for word in cleaned_text.split()):
        return "Negative"
    return "Neutral"

class MarkovNLG:
    """[SYLLABUS: NLG with Markov]"""
    def __init__(self):
        self.chain = {}
    
    def train(self, corpus):
        words = corpus.split()
        for i in range(len(words)-1):
            if words[i] not in self.chain:
                self.chain[words[i]] = []
            self.chain[words[i]].append(words[i+1])
            
    def generate(self, start_word, length=10):
        if start_word not in self.chain:
            return start_word
        current = start_word
        result = [current]
        for _ in range(length):
            next_words = self.chain.get(current)
            if not next_words:
                break
            current = random.choice(next_words)
            result.append(current)
            if current.endswith('.'): # Stop at sentence end
                break
        return ' '.join(result)

# Train TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(st.session_state.kb_df['Sample_Queries'])

def get_chatbot_response(user_input):
    log_chat(user_input) # Save to log for Topic Modeling
    cleaned_input = clean_rojak_text(user_input)
    
    sentiment = analyze_sentiment(cleaned_input)
    pos_tags, wsd_tags = perform_pos_and_wsd(cleaned_input) # Execute POS & WSD
    
    # [SYLLABUS: TF-IDF & Semantic matching]
    input_vector = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix)
    best_match_index = similarities.argmax()
    confidence_score = similarities[0][best_match_index]
    
    if confidence_score < 0.15:
        return "Sorry lah, I didn't quite catch that. Can you rephrase?", "unknown", sentiment, pos_tags
    
    intent = st.session_state.kb_df['Intent'].iloc[best_match_index]
    response_variations = st.session_state.kb_df['Response_Variations'].iloc[best_match_index]
    
    # Run Markov NLG to generate a dynamic response based on variations
    markov = MarkovNLG()
    markov.train(response_variations.replace('|', ' '))
    base_response = random.choice(response_variations.split('|')) # Fallback
    generated_reply = markov.generate(base_response.split()[0], length=12)
    
    if sentiment == "Negative":
        generated_reply = "I understand you are frustrated. Let me help. " + generated_reply
        
    return generated_reply, intent, sentiment, pos_tags

# ==========================================
# 3. STREAMLIT UI & ADMIN DASHBOARD
# ==========================================

st.set_page_config(page_title="Rojak Support Bot", layout="centered")
st.title("📦 Rojak E-Com Support")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎙️ Audio", "📊 Admin (LDA & NLP Logic)"])
user_query = None

with tab1:
    text_input = st.text_input("Type your issue:", placeholder="e.g., Asal barang I tak sampai?")
    if text_input:
        user_query = text_input

with tab2:
    recognizer = sr.Recognizer()
    audio_bytes = st.audio_input("Speak to customer service:")
    if audio_bytes:
        with st.spinner("Translating..."):
            try:
                audio_file = io.BytesIO(audio_bytes.read())
                with sr.AudioFile(audio_file) as source:
                    user_query = recognizer.recognize_google(recognizer.record(source), language="ms-MY")
            except Exception as e:
                st.error("Audio error. Try typing instead.")

# CHAT LOGIC & FEEDBACK LOOP
if user_query:
    st.session_state.last_query = user_query
    response, intent, sentiment, pos_tags = get_chatbot_response(user_query)
    st.session_state.last_response = response
    st.session_state.last_intent = intent
    st.session_state.pos_tags = pos_tags
    st.session_state.show_feedback = True

if st.session_state.get('show_feedback', False):
    st.info(f"**You:** {st.session_state.last_query}")
    st.success(f"**Bot ({st.session_state.last_intent}):** {st.session_state.last_response}")
    
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Correct!"):
            st.success("Thanks!")
            st.session_state.show_feedback = False
    with col2:
        if st.button("👎 Wrong Intent"):
            st.session_state.show_correction = True
            
    if st.session_state.get('show_correction', False):
        correct_intent = st.selectbox("What did you actually mean?", st.session_state.kb_df['Intent'].unique())
        if st.button("Submit Fix"):
            # Update Knowledge Base CSV persistently
            correct_response = st.session_state.kb_df[st.session_state.kb_df['Intent'] == correct_intent]['Response_Variations'].iloc[0]
            save_to_kb(correct_intent, clean_rojak_text(st.session_state.last_query), correct_response)
            st.success("Database updated! I will remember this.")
            st.session_state.show_correction = False
            st.session_state.show_feedback = False

# [SYLLABUS: Topic Modeling LDA]
with tab3:
    st.header("Admin Dashboard (NLP Analytics)")
    if st.session_state.get('pos_tags'):
        st.write("**Latest Syntactic Analysis (POS Tagging):**", st.session_state.pos_tags)
        
    st.write("---")
    st.subheader("Trending Topics (LDA)")
    if os.path.exists(LOG_FILE):
        logs_df = pd.read_csv(LOG_FILE)
        if len(logs_df) > 3:
            lda_vectorizer = CountVectorizer(stop_words='english')
            dtm = lda_vectorizer.fit_transform(logs_df['query'].apply(clean_rojak_text))
            lda = LatentDirichletAllocation(n_components=2, random_state=42)
            lda.fit(dtm)
            
            for index, topic in enumerate(lda.components_):
                top_words = [lda_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
                st.write(f"**Topic {index+1}:** {', '.join(top_words)}")
        else:
            st.write("Not enough chat logs to run LDA Topic Modeling yet. Chat with the bot a few more times!")
    else:
        st.write("No chat logs found.")