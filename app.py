import asyncio
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from collections import Counter
import re
import string
import urllib.request
import os

try:
    from pyvi import ViTokenizer
except ImportError:
    pass

import gemini_sentiment
from data_preprocessing import preprocess_threads_comments
from gemini_sentiment import analyze_comments
from threads.threads_runner import comments_to_dataframe, get_threads_data, posts_to_dataframe


st.set_page_config(page_title="Social Sentiment App", layout="wide")

DATA_DIR = Path("generated")
DATA_DIR.mkdir(exist_ok=True)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(255, 196, 140, 0.22), transparent 30%),
            radial-gradient(circle at top left, rgba(94, 234, 212, 0.18), transparent 28%),
            linear-gradient(180deg, #f7f3ea 0%, #fbfaf7 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #172033 0%, #21485f 52%, #f0a35e 130%);
        color: #fff9f1;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 18px 50px rgba(23, 32, 51, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0.6rem 0 0 0;
        color: rgba(255, 249, 241, 0.9);
        font-size: 1rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(23, 32, 51, 0.08);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 30px rgba(23, 32, 51, 0.06);
    }
    .section-note {
        color: #52606d;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SENTIMENT_ORDER  = ["positive", "neutral", "negative", "mixed"]
SENTIMENT_COLORS = ["#2BB673", "#7B8794", "#E45756", "#F0A35E"]
EMOTION_COLORS   = ["#ff9f1c", "#577590", "#d62828", "#7b2cbf", "#2a9d8f", "#8d99ae"]
TOPIC_COLORS     = ["#15616d", "#1d7874", "#679289", "#f4a261", "#e76f51",
                    "#7f5539", "#6d597a", "#355070", "#457b9d", "#e9c46a"]


def run_async(coro):
    return asyncio.run(coro)


def set_runtime_api_key(api_keys_str: str):
    # Thay dấu phẩy bằng dấu xuống dòng để hỗ trợ cả 2 cách nhập
    api_keys_str = api_keys_str.replace(",", "\n")
    keys = [k.strip() for k in api_keys_str.split("\n") if k.strip()]
    if keys:
        gemini_sentiment.GEMINI_API_KEYS = keys
        gemini_sentiment.current_key_index = 0


def build_metric_cards(df: pd.DataFrame):
    if df.empty:
        return [
            ("So comment", "0"),
            ("Confidence TB", "0.00"),
            ("Ti le controversy", "0.0%"),
            ("Sentiment troi", "-"),
        ]
    dominant_sentiment = (
        df["sentiment"].mode().iloc[0]
        if "sentiment" in df.columns and not df["sentiment"].dropna().empty
        else "-"
    )
    confidence_mean   = df["confidence"].fillna(0).mean() if "confidence" in df.columns else 0
    controversy_rate  = df["controversy"].fillna(False).mean() if "controversy" in df.columns else 0
    return [
        ("So comment",        f"{len(df):,}"),
        ("Confidence TB",     f"{confidence_mean:.2f}"),
        ("Ti le controversy", f"{controversy_rate * 100:.1f}%"),
        ("Sentiment troi",    dominant_sentiment),
    ]


def render_metric_cards(df: pd.DataFrame):
    cols = st.columns(4)
    for col, (label, value) in zip(cols, build_metric_cards(df)):
        col.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.9rem;color:#667085;">{label}</div>
                <div style="font-size:1.7rem;font-weight:700;color:#172033;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_download_button(df: pd.DataFrame, label: str, file_name: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )



@st.cache_data
def load_stopwords():
    stopword_file = 'threads/vietnamese-stopwords.txt'
    if not os.path.exists(stopword_file):
        try:
            url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
            urllib.request.urlretrieve(url, stopword_file)
        except Exception:
            pass
            
    stopwords_vn = set()
    if os.path.exists(stopword_file):
        with open(stopword_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().replace(' ', '_')
                if word:
                    stopwords_vn.add(word)
    
    user_custom_stopwords = {
        "là", "và", "thì", "mà", "đó", "này", "đấy", "rồi", "nha", "ha", "ờ", "ừ", 
        "à", "ơ", "ê", "ừm", "vậy", "nếu", "sao", "xong", "giờ", "kiểu", "cái", "còn", 
        "đến", "đi", "được", "người", "bạn", "tụi", "tụi_mày", "mấy", "một", "hai", 
        "ba", "đang", "nữa", "quá", "thật", "rất", "luôn", "chỉ", "nên", "cũng", 
        "với", "cho", "trong", "ngoài", "về", "khi", "lúc", "tại", "do", "có", "bị", 
        "làm", "thấy", "biết", "nghe", "xem", "nói", "đọc", "ơi", "con", "ổng", "bả", 
        "ảnh", "chị", "anh", "em", "tao", "mày", "tui", "t", "tụi_bây",
        "nhá", "ạ", "á", "hả", "kìa", "của"
    }
    stopwords_vn.update(user_custom_stopwords)
    return stopwords_vn

def clean_text(text, stopwords_vn):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    try:
        text = ViTokenizer.tokenize(text)
    except NameError:
        pass
    words = text.split()
    words = [w for w in words if w not in stopwords_vn and len(w) > 1]
    return words

def render_viz_dashboard(df: pd.DataFrame):
    if df.empty:
        st.info("Chưa có dữ liệu để trực quan.")
        return
        
    color_map = {
        'positive': '#00CC96',
        'negative': '#EF553B',
        'neutral': '#636EFA',
        'mixed': '#FFA15A',
        'unknown': '#B6E880'
    }

    if 'controversy' in df.columns:
        df['controversy'] = df['controversy'].fillna(False).astype(bool)

    st.markdown("---")
    st.markdown("## SECTION A: OVERVIEW & DISTRIBUTIONS")
    
    colA1, colA2 = st.columns(2)
    with colA1:
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index, 
                title='A. Phân bố cảm xúc (Sentiment)',
                hole=0.4,
                color=sentiment_counts.index,
                color_discrete_map=color_map
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    with colA2:
        if 'emotion' in df.columns:
            emotion_counts = df['emotion'].value_counts().head(12)
            fig = px.bar(
                x=emotion_counts.values, 
                y=emotion_counts.index, 
                orientation='h',
                title='A. Emotion xuất hiện nhiều nhất',
                labels={'x': 'Count', 'y': 'Emotion'},
                color=emotion_counts.values,
                color_continuous_scale='Magma'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
    colA3, colA4 = st.columns(2)
    with colA3:
        if 'topic' in df.columns:
            topic_counts = df['topic'].value_counts().head(12)
            fig = px.treemap(
                names=topic_counts.index,
                parents=[""]*len(topic_counts),
                values=topic_counts.values,
                title='A. Topic được nhắc nhiều',
                color=topic_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with colA4:
        if 'controversy' in df.columns:
            controversy_counts = df['controversy'].value_counts()
            controversy_rate = (controversy_counts.get(True, 0) / len(df)) * 100
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = controversy_rate,
                title = {'text': "A. Tỷ lệ tranh cãi (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF553B"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## SECTION B: CROSS-ANALYSIS")
    
    colB1, colB2 = st.columns(2)
    with colB1:
        if 'topic' in df.columns and 'sentiment' in df.columns:
            top_topics = df['topic'].value_counts().head(10).index
            topic_sentiment = pd.crosstab(df[df['topic'].isin(top_topics)]['topic'], df['sentiment'], normalize='index') * 100
            fig = px.bar(
                topic_sentiment.reset_index(),
                x='topic',
                y=topic_sentiment.columns,
                title='B. Sentiment rate theo Topic',
                labels={'value': 'Percentage (%)', 'topic': 'Topic', 'sentiment': 'Sentiment'},
                color_discrete_map=color_map
            )
            fig.update_layout(barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
            
    with colB2:
        if 'emotion' in df.columns and 'controversy' in df.columns:
            emotion_controversy = df.groupby('emotion')['controversy'].mean().sort_values(ascending=False).head(10) * 100
            fig = px.bar(
                x=emotion_controversy.values,
                y=emotion_controversy.index,
                orientation='h',
                title='B. Controversy rate theo Emotion',
                labels={'x': 'Controversy Rate (%)', 'y': 'Emotion'},
                color=emotion_controversy.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
    colB3, _ = st.columns([1, 1])
    with colB3:
        if 'topic' in df.columns and 'controversy' in df.columns:
            topic_controversy = df.groupby('topic')['controversy'].mean().sort_values(ascending=False).head(10) * 100
            fig = px.bar(
                x=topic_controversy.values,
                y=topic_controversy.index,
                orientation='h',
                title='B. Controversy rate theo Topic',
                labels={'x': 'Controversy Rate (%)', 'y': 'Topic'},
                color=topic_controversy.values,
                color_continuous_scale='Oranges'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("---")
    st.markdown("## SECTION C: TEXT ANALYSIS")
    
    stopwords_vn = load_stopwords()
    # Create a copy so we don't mutate original dataframe multiple times
    df_temp = df.copy()
    df_temp['cleaned_words'] = df_temp['comment_text'].apply(lambda x: clean_text(x, stopwords_vn))
    all_words = [word for words in df_temp['cleaned_words'] for word in words]
    
    total_words = len(all_words)
    unique_words = len(set(all_words))
    avg_length = df_temp['cleaned_words'].apply(len).mean()
    
    colC1, colC2, colC3 = st.columns(3)
    colC1.metric("Total Words", f"{total_words:,}")
    colC2.metric("Unique Words", f"{unique_words:,}")
    colC3.metric("Avg Words/Comment", f"{avg_length:.1f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    colC4, colC5 = st.columns(2)
    with colC4:
        word_counts = Counter(all_words)
        top_words = dict(word_counts.most_common(20))
        if top_words:
            fig = px.bar(
                x=list(top_words.values()),
                y=list(top_words.keys()),
                orientation='h',
                title='C. Top Word Frequency',
                labels={'x': 'Frequency', 'y': 'Word'},
                color=list(top_words.values()),
                color_continuous_scale='Teal'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
    with colC5:
        try:
            from wordcloud import WordCloud
            text_for_cloud = " ".join(all_words)
            if text_for_cloud.strip():
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab20').generate(text_for_cloud)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                ax.set_title('C. Word Cloud')
                st.pyplot(fig_wc)
        except ImportError:
            st.warning("Cần cài đặt thư viện wordcloud để hiển thị biểu đồ này.")
            
    colC6, colC7 = st.columns(2)
    with colC6:
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            corpus_for_bigram = df_temp['cleaned_words'].apply(lambda x: ' '.join(x))
            if any(corpus_for_bigram.str.strip() != ""):
                vectorizer = CountVectorizer(ngram_range=(2, 2))
                X = vectorizer.fit_transform(corpus_for_bigram)
                bigram_counts = pd.DataFrame(
                    {'bigram': vectorizer.get_feature_names_out(), 'count': X.sum(axis=0).A1}
                ).sort_values(by='count', ascending=False).head(15)
                fig = px.bar(
                    bigram_counts,
                    x='count',
                    y='bigram',
                    orientation='h',
                    title='C. Bigram Analysis',
                    labels={'count': 'Frequency', 'bigram': 'Bigram'},
                    color='count',
                    color_continuous_scale='Purples'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Cần cài đặt thư viện scikit-learn để hiển thị phân tích Bigram.")
            
    with colC7:
        if 'sentiment' in df_temp.columns:
            pos_words = [word for words in df_temp[df_temp['sentiment'] == 'positive']['cleaned_words'] for word in words]
            neg_words = [word for words in df_temp[df_temp['sentiment'] == 'negative']['cleaned_words'] for word in words]

            top_pos = dict(Counter(pos_words).most_common(10))
            top_neg = dict(Counter(neg_words).most_common(10))

            if top_pos or top_neg:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Top Words (Positive)", "Top Words (Negative)"))
                if top_pos:
                    fig.add_trace(
                        go.Bar(x=list(top_pos.values())[::-1], y=list(top_pos.keys())[::-1], orientation='h', marker_color='#00CC96', name="Positive"),
                        row=1, col=1
                    )
                if top_neg:
                    fig.add_trace(
                        go.Bar(x=list(top_neg.values())[::-1], y=list(top_neg.keys())[::-1], orientation='h', marker_color='#EF553B', name="Negative"),
                        row=1, col=2
                    )

                fig.update_layout(
                    title='C. Sentiment Word Analysis',
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_sample_table(df: pd.DataFrame, text_column: str):
    preferred_cols = [text_column, "sentiment", "emotion", "topic", "controversy", "confidence", "rationale"]
    cols = [col for col in preferred_cols if col in df.columns]
    st.dataframe(df[cols].head(12), use_container_width=True)


def run_threads_pipeline(queries, post_limit, comments_per_post, celebrity_name, batch_size, limit):
    posts, comments = run_async(
        get_threads_data(
            queries=queries,
            post_limit_per_query=post_limit,
            comments_per_post=comments_per_post,
        )
    )
    posts_df         = posts_to_dataframe(posts)
    comments_df      = comments_to_dataframe(comments)
    comments_clean_df = preprocess_threads_comments(comments_df)
    sentiment_df     = analyze_comments(
        comments_clean_df,
        text_column="comment_text",
        celebrity_name=celebrity_name,
        batch_size=batch_size,
        limit=limit,
    )
    return posts_df, comments_df, comments_clean_df, sentiment_df


# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>Social Sentiment Studio</h1>
        <p>Crawl Threads, tien xu ly comment, phan tich cam xuc bang Gemini.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    
    with st.expander("🔑 Hướng dẫn lấy Gemini API Key miễn phí", expanded=False):
        st.markdown(
            """
            1. Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey).
            2. Đăng nhập bằng tài khoản Google của bạn.
            3. Nhấn nút **Create API key** để tạo khóa.
            4. Copy mã vừa tạo và dán vào ô bên dưới.
            
            👉 *Mẹo: Bạn có thể tạo nhiều API Key từ nhiều tài khoản Google khác nhau để không bị ngắt quãng. Hãy copy và dán mỗi key trên 1 dòng vào ô dưới đây.*
            """
        )

    runtime_api_key = st.text_area(
        "Nhập danh sách Gemini API Key", 
        placeholder="Mỗi API Key nằm trên 1 dòng:\nAIzaSy_123...\nAIzaSy_456...",
        height=100
    )
    if runtime_api_key:
        set_runtime_api_key(runtime_api_key)
        st.success(f"✅ Đã nạp danh sách API Key. Hệ thống sẵn sàng!")
    else:
        st.info("💡 Chưa nhập API key. Hệ thống sẽ tự động dùng key mặc định nếu có trong code.")

    batch_size = st.number_input("So comment trong 1 lan goi Gemini", min_value=1, max_value=50, value=5)
    analyze_all_comments = st.checkbox("Phan tich toan bo comment crawl duoc", value=True)
    limit = None
    if not analyze_all_comments:
        limit = int(st.number_input("Gioi han tong so comment dua vao LLM", min_value=1, max_value=5000, value=100))

    st.caption("Threads can Playwright va co the can session dang nhap hop le.")


# ─── Tab Threads ──────────────────────────────────────────────────────────────

st.subheader("Threads Sentiment")
st.markdown(
    '<div class="section-note">Nhap ten nghe si va bien the query. App se tim, loc post, lay comment, roi phan tich sentiment.</div>',
    unsafe_allow_html=True,
)

col_form_1, col_form_2, col_form_3 = st.columns([2, 1, 1])
with col_form_1:
    threads_query_input = st.text_input("Ten nghe si cho Threads", value="Miu Le, miule")
with col_form_2:
    threads_post_count = st.number_input("So bai post lay", min_value=1, max_value=50, value=2)
with col_form_3:
    threads_comment_count = st.number_input("So cmt lay moi bai", min_value=1, max_value=200, value=20)

if st.button("Chay Threads", type="primary", use_container_width=True):
    queries = [item.strip() for item in threads_query_input.split(",") if item.strip()]
    if not queries:
        st.error("Can nhap it nhat 1 query cho Threads.")
    else:
        with st.spinner("Dang crawl va phan tich Threads..."):
            try:
                posts_df, comments_df, comments_clean_df, sentiment_df = run_threads_pipeline(
                    queries=queries,
                    post_limit=int(threads_post_count),
                    comments_per_post=int(threads_comment_count),
                    celebrity_name=queries[0],
                    batch_size=int(batch_size),
                    limit=limit,
                )
                st.session_state["threads_raw_posts"]     = posts_df
                st.session_state["threads_raw_comments"]  = comments_df
                st.session_state["threads_clean_comments"] = comments_clean_df
                st.session_state["threads_sentiment"]     = sentiment_df
                st.success("Da xu ly xong Threads.")
            except Exception as exc:
                st.exception(exc)

if "threads_sentiment" in st.session_state:
    render_metric_cards(st.session_state["threads_sentiment"])
    st.markdown("### Truc quan sentiment")
    render_viz_dashboard(st.session_state["threads_sentiment"])
    st.markdown("### Mau ket qua")
    render_sample_table(st.session_state["threads_sentiment"], "comment_text")

    with st.expander("Bang post raw"):
        st.dataframe(st.session_state["threads_raw_posts"], use_container_width=True)
    with st.expander("Bang comment clean"):
        st.dataframe(st.session_state["threads_clean_comments"], use_container_width=True)
    with st.expander("Bang sentiment day du"):
        st.dataframe(st.session_state["threads_sentiment"], use_container_width=True)

    render_download_button(
        st.session_state["threads_sentiment"],
        "Tai sentiment Threads CSV",
        "threads_sentiment.csv",
    )
