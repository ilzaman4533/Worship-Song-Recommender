import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import re

st.set_page_config(page_title="Worship Song Recommender", layout="centered")

@st.cache_resource
def load_resources():
    df = pd.read_csv('Worship/data/worship_songs.csv')
    df['search_text'] = df['speed'] + ' ' + df['themes'] + ' ' + df['title'] + ' ' + df['artist']
    
    bi_encoder = SentenceTransformer('all-mpnet-base-v2')
    embeddings = bi_encoder.encode(df['search_text'].tolist(), convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    return df, bi_encoder, embeddings, index, cross_encoder

df, bi_encoder, embeddings, index, cross_encoder = load_resources()

def extract_speed_filter(query):
    query_lower = query.lower()
    if re.search(r"\b(slow|slower|slowly)\b", query_lower):
        return "slow"
    elif re.search(r"\b(mid|middle|medium|moderate|mid-tempo|midtempo)\b", query_lower):
        return "middle"
    elif re.search(r"\b(fast|faster|quick|upbeat)\b", query_lower):
        return "fast"
    return None

def recommend(query, top_k=20, candidate_pool=len(df)):
    speed_filter = extract_speed_filter(query)

    filtered_df = df
    if speed_filter:
        filtered_df = df[df['speed'].str.lower() == speed_filter]
        if filtered_df.empty:
            return pd.DataFrame()

    query_emb = bi_encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    search_texts = filtered_df['search_text'].tolist()
    local_embeddings = bi_encoder.encode(search_texts, convert_to_numpy=True)
    faiss.normalize_L2(local_embeddings)

    temp_index = faiss.IndexFlatIP(local_embeddings.shape[1])
    temp_index.add(local_embeddings)
    distances, candidate_idxs = temp_index.search(query_emb, candidate_pool)

    candidates = filtered_df.iloc[candidate_idxs[0]].copy()

    cross_inputs = [
        (query, row['speed'] + " " + row['themes'] + " " + row['lyrics']) 
        for _, row in candidates.iterrows()
    ]
    cross_scores = cross_encoder.predict(cross_inputs)

    candidates['score'] = cross_scores

    # Stable sort to prevent shuffling or duplicate views
    candidates = candidates.sort_values(by=['score', 'title'], ascending=[False, True]).head(top_k)

    return candidates.reset_index(drop=True)


# --- UI Styling ---
st.markdown("""
    <style>
    .responsive-container {
        max-width: 700px;
        margin: auto;
        padding: 10px;
    }
    .song-card {
        background-color: #f9f9f9;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
        color: #222;
        word-wrap: break-word;
    }
    .song-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2e3b55;
        margin-bottom: 4px;
    }
    .song-meta {
        font-size: 0.95rem;
        margin-bottom: 8px;
    }
    .song-link {
        font-size: 0.9rem;
        color: #007acc;
    }
    @media (max-width: 768px) {
        .song-title {
            font-size: 1rem;
        }
        .song-meta {
            font-size: 0.85rem;
        }
        .song-card {
            padding: 14px;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='responsive-container'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🎵 Worship Song Recommender</h1>", unsafe_allow_html=True)
st.write("Enter themes, phrases, or song speed like **'slow'**, **'middle'**, or **'fast'** to get worship song suggestions.")

query = st.text_input("🔍 What are you looking for?")

if 'visible_count' not in st.session_state:
    st.session_state.visible_count = 5
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

if query and query != st.session_state.last_query:
    st.session_state.results = recommend(query)
    st.session_state.visible_count = 5
    st.session_state.last_query = query

results = st.session_state.results
visible_count = st.session_state.visible_count

if query:
    if results.empty:
        st.warning("No matching songs found. Try adjusting your query or speed filter.")
    else:
        st.write(f"🎧 Showing top {min(visible_count, len(results))} of {len(results)} results:")

        pastel_colors = [
            "#e0f7fa", "#ffe0b2", "#f3e5f5", "#e1f5fe", "#fff9c4",
            "#dcedc8", "#f8bbd0", "#d1c4e9", "#fbe9e7", "#e6ee9c"
        ]

        for idx, (_, row) in enumerate(results.iloc[:visible_count].iterrows()):
            bg_color = pastel_colors[idx % len(pastel_colors)]
            st.markdown(f"""
                <div class="song-card" style="background-color: {bg_color};">
                    <div class="song-title">🎶 {row['title']} <span style='font-weight:normal;'>– {row['artist']}</span></div>
                    <div class="song-meta"><strong>Score:</strong> {row['score']:.4f}</div>
                    <div class="song-meta"><strong>Themes:</strong> {row['themes']}</div>
                    <div class="song-meta"><strong>Speed:</strong> {row['speed'].capitalize()}</div>
                    <a class="song-link" href="{row['pnwchords_link']}" target="_blank">🔗 View on PNWChords</a>
                </div>
            """, unsafe_allow_html=True)

        if visible_count < 10:
            if st.button("🎵 See More"):
                st.session_state.visible_count += 1

else:
    st.info("Type a query to start finding songs.")

st.markdown("</div>", unsafe_allow_html=True)

