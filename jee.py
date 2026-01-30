Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import googletrans  # pip install googletrans==4.0.0rc1

# Custom Hindi translator (simplified)
def translate_to_hindi(text):
    """Simple Hindi translations for common recommendations"""
    hindi_map = {
        "Focus on conceptual understanding": "अवधारणा समझ पर ध्यान दें",
        "Practice more numerical problems": "अधिक संख्यात्मक प्रश्नों का अभ्यास करें",
        "Improve time management": "समय प्रबंधन सुधारें",
        "Revise basic formulas": "मूल सूत्रों का पुनरावलोकन करें",
        "Work on accuracy": "सटीकता पर काम करें",
        "Solve previous year questions": "पिछले वर्षों के प्रश्न हल करें"
    }
    return hindi_map.get(text, text)

# Page config
st.set_page_config(page_title="JEE Analytics Pro", layout="wide", page_icon="📊")

st.title("🚀 JEE Mock Test AI Analytics")
st.markdown("Upload your JEE mock test CSV for question-level insights & personalized Hindi recommendations")

# File uploader
uploaded_file = st.file_uploader("Choose JEE Mock CSV", type=['csv'], 
                                help="Expected columns: question_id, subject, topic, marks, time_spent, correct (1/0), mistake_type")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} questions!")
    
    # Data cleaning
    df['marks'] = pd.to_numeric(df['marks'], errors='coerce').fillna(0)
    df['time_spent'] = pd.to_numeric(df['time_spent'], errors='coerce').fillna(0)
    df['correct'] = df['correct'].astype(int)
    df['accuracy'] = df['correct']
    df['error'] = 1 - df['correct']
    
    # Sidebar metrics
    st.sidebar.header("📈 Quick Stats")
    col1, col2, col3, col4 = st.sidebar.columns(4)
    with col1: st.metric("Total Score", f"{df['marks'].sum():.0f}/300")
    with col2: st.metric("Accuracy", f"{df['accuracy'].mean():.1%}")
    with col3: st.metric("Avg Time/Q", f"{df['time_spent'].mean():.0f}s")
    with col4: st.metric("Weak Topics", len(df[df['correct']==0]['topic'].unique()))
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Error Clusters", "📚 Recommendations", "📈 Progress"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Subject Breakdown")
            subject_df = df.groupby('subject')[['marks', 'accuracy']].agg(['sum', 'mean']).round(2)
            st.dataframe(subject_df, use_container_width=True)
            
            # Subject chart
            fig_subject = px.bar(df.groupby('subject')['accuracy'].mean().reset_index(), 
                               x='subject', y='accuracy', title="Accuracy by Subject")
            st.plotly_chart(fig_subject, use_container_width=True)
        
        with col2:
            st.subheader("Weakest Topics")
            weak_topics = df[df['correct']==0]['topic'].value_counts().head(8)
            st.bar_chart(weak_topics)
    
    with tab2:
        st.header("🤖 AI Error Clustering")
        
        # Prepare clustering data
        cluster_features = ['time_spent', 'marks', 'error']
        X = df[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering (4 clusters: Easy/Wrong, Hard/Correct, Time Waste, etc.)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        st.subheader("Your 4 Error Patterns")
        cluster_names = {
            0: "🚫 Conceptual Gaps", 1: "⏱️ Time Wasters", 
            2: "✅ Good but Slow", 3: "❌ Careless Errors"
        }
        
        cluster_summary = df.groupby('cluster')[cluster_features].agg(['mean', 'count']).round(2)
        cluster_summary['count'] = df['cluster'].value_counts().sort_index()
        st.dataframe(cluster_summary)
        
        # Cluster visualization
        fig_clusters = px.scatter(df, x='time_spent', y='marks', 
                                color='cluster', size='error',
                                hover_data=['topic', 'correct'],
                                title="Error Patterns (Clusters)")
        st.plotly_chart(fig_clusters, use_container_width=True)
    
    with tab3:
        st.header("💡 Personalized Study Plan")
        
        # Generate recommendations based on clusters
        recommendations = []
        
        # Cluster-based insights
        if df['cluster'].value_counts().get(1, 0) > 5:  # Time wasters
            recommendations.append("Improve time management")
        if df['cluster'].value_counts().get(0, 0) > 8:  # Conceptual gaps
            recommendations.append("Focus on conceptual understanding")
        if (df['accuracy'].mean() < 0.6) and (df['time_spent'].mean() > 120):
            recommendations.append("Practice more numerical problems")
        
        # Topic-specific
        weak_topic = df[df['correct']==0]['topic'].value_counts().index[0]
        recommendations.append(f"Revise {weak_topic}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("English Recommendations")
            for rec in recommendations:
...                 st.info(f"• {rec}")
...         
...         with col2:
...             st.subheader("हिंदी सिफारिशें (Hindi)")
...             for rec in recommendations:
...                 st.info(f"• {translate_to_hindi(rec)}")
...         
...         # Action plan
...         st.markdown("---")
...         st.subheader("✅ Next 7 Days Action Plan")
...         st.markdown("""
...         1. **Day 1-2**: Revise {weak_topic} (20 PYQs)
...         2. **Day 3-4**: Timed practice (same topics, 90 mins)
...         3. **Day 5**: Full topic test
...         4. **Day 6-7**: Review mistakes + formula sheet
...         """.format(weak_topic=weak_topic))
...     
...     with tab4:
...         st.header("📊 Track Improvement")
...         st.info("💡 Pro Tip: Run this analysis after every mock to see progress!")
...         
...         # Mock progress simulation (for demo)
...         progress_data = pd.DataFrame({
...             'Mock': range(1, 6),
...             'Score': [120, 145, 167, 189, 215],
...             'Accuracy': [0.45, 0.52, 0.58, 0.65, 0.72]
...         })
...         
...         fig_progress = make_subplots(specs=[[{"secondary_y": True}]])
...         fig_progress.add_trace(
...             go.Scatter(x=progress_data['Mock'], y=progress_data['Score'], 
...                       name="Score", line=dict(color='blue')), secondary_y=False)
...         fig_progress.add_trace(
...             go.Scatter(x=progress_data['Mock'], y=progress_data['Accuracy'], 
...                       name="Accuracy", line=dict(color='green')), secondary_y=True)
...         fig_progress.update_layout(title="Expected Progress Following Recommendations")
...         st.plotly_chart(fig_progress)
... 
... # Footer
... st.markdown("---")
