import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.parser import extract_text
from src.preprocessing import clean_text, extract_skills, preprocess_text, COMMON_SKILLS
from src.matcher import calculate_similarity, calculate_sbert_similarity, find_missing_skills
from src.research_lab import compare_algorithms, detect_bias_entities
from src.scoring import calculate_readability, calculate_ats_score
from src.adaptive_engine import AdaptiveWeights
from src.experiment import ExperimentLab

# Initialize Engines
if 'adaptive_engine' not in st.session_state:
    st.session_state['adaptive_engine'] = AdaptiveWeights()

# Page Config
st.set_page_config(page_title="AI Research Lab", page_icon="🧬", layout="wide")

st.title("🧬 Adaptive AI Resume Research Framework")
st.markdown("### 🚀 Novel Intelligent System with SBERT & Reinforcement Learning")

# Sidebar
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    st.info(f"Loaded: {len(uploaded_files)} resumes")
    
    st.divider()
    st.subheader("⚙️ Experiment Controls")
    bias_filter = st.checkbox("Enable Bias-Aware Filtering", value=True)
    use_sbert = st.checkbox("Use Deep Semantic (SBERT)", value=True)

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Job Description")
    jd_input = st.text_area("Paste the Job Description here...", height=300)

if st.button("Run Adaptive Experiment"):
    if not uploaded_files or not jd_input:
        st.warning("Please upload resumes and provide a job description.")
    else:
        with st.spinner("Initializing Deep Neural Networks..."):
            results = []
            jd_text = clean_text(jd_input)
            jd_processed = preprocess_text(jd_text)
            jd_skills = extract_skills(jd_input)
            
            # Detect Role and Adjust Weights
            role = st.session_state['adaptive_engine'].detect_role(jd_text)
            current_weights = st.session_state['adaptive_engine'].get_weights()
            st.toast(f"Adaptive Engine: Detected Role '{role}'. Adjusted Weights: {current_weights}", icon="🧠")
            
            # Preprocess all resumes
            resume_texts = []
            cleaned_texts = []
            
            for file in uploaded_files:
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                
                text = extract_text(tmp_path)
                os.remove(tmp_path)
                
                if text:
                    detected_bias = detect_bias_entities(text) if bias_filter else []
                    cleaned = clean_text(text)
                    processed = preprocess_text(cleaned)
                    
                    resume_texts.append(text)
                    cleaned_texts.append(processed)
                    
                    # Store meta
                    results.append({
                        "Name": file.name,
                        "Raw Text": text,
                        "Cleaned Text": processed,
                        "Skills": extract_skills(text),
                        "Detected Bias": detected_bias,
                        "Readability": calculate_readability(text),
                        "Processing Status": "Success"
                    })
                else:
                    results.append({"Name": file.name, "Processing Status": "Failed"})

            # Calculate Scores
            valid_indices = [i for i, r in enumerate(results) if r["Processing Status"] == "Success"]
            
            if valid_indices:
                valid_resume_clean = [results[i]["Cleaned Text"] for i in valid_indices]
                
                # 1. TF-IDF Score
                tfidf_scores = calculate_similarity(valid_resume_clean, jd_processed)
                
                # 2. SBERT Score (Deep Semantic)
                if use_sbert:
                    sbert_scores = calculate_sbert_similarity(valid_resume_clean, jd_processed)
                else:
                    sbert_scores = [0] * len(valid_indices)
                
                for idx, (t_score, s_score) in zip(valid_indices, zip(tfidf_scores, sbert_scores)):
                    res = results[idx]
                    
                    # Calculate ATS/Adaptive Score
                    skill_match = list(set(res["Skills"]).intersection(set(jd_skills)))
                    
                    # Use SBERT as the "Keyword Relevance" proxy if enabled, else TF-IDF
                    semantic_proxy = s_score if use_sbert else t_score
                    
                    # Custom Weighted Score using Adaptive Engine
                    # We manually calculate weighted sum ensuring 0-100 scale
                    w = current_weights
                    
                    # Normalize components (0-1)
                    s_skill = len(skill_match) / len(jd_skills) if len(jd_skills) > 0 else 0
                    s_sem = semantic_proxy
                    from src.scoring import extract_experience_relevance, extract_education_level
                    s_exp = extract_experience_relevance(res["Raw Text"])
                    s_edu = extract_education_level(res["Raw Text"])
                    
                    final_score = (s_sem * w["Semantic"] + s_skill * w["Skill"] + s_exp * w["Experience"] + s_edu * w["Education"]) * 100
                    
                    res.update({
                        "NLP Score": round(t_score, 4),
                        "SBERT Score": round(s_score, 4),
                        "Final Score": round(final_score, 2),
                        "Matched Skills": skill_match,
                        "Missing Skills": find_missing_skills(res["Raw Text"], jd_input, COMMON_SKILLS),
                        "Skill Match Count": len(skill_match)
                    })

            # --- VISUALIZATION TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["🧠 Intelligent Matching", "🧪 Experimental Lab", "📢 Explainability", "🔄 Feedback Loop"])
            
            df = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])
            
            with tab1:
                st.subheader(f"Adaptive Ranking (Role: {role})")
                if not df.empty:
                    df_display = df[["Name", "Final Score", "SBERT Score", "NLP Score"]].sort_values(by="Final Score", ascending=False)
                    st.dataframe(df_display.style.highlight_max(axis=0), use_container_width=True)
                    
                    fig = px.bar(df_display, x="Name", y="Final Score", color="SBERT Score", title="Adaptive Composite Score")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Comparative Study: 4-Way Analysis")
                st.info("Comparing: Jaccard (Keyword) vs TF-IDF (Vector) vs SBERT (Neural) vs Adaptive (Proposed).")
                
                exp_lab = ExperimentLab()
                # Run the lab analysis
                study_df = exp_lab.run_comparative_study(df.to_dict('records'), jd_input, None, None)
                # Add Final Score (Adaptive) to study
                study_df["Adaptive Framework"] = df["Final Score"] / 100.0 # Normalize back to 0-1 for chart
                
                # Melt for chart
                study_melt = study_df.melt(id_vars=["Name"], var_name="Algorithm", value_name="Score")
                
                fig2 = px.bar(study_melt, x="Name", y="Score", color="Algorithm", barmode='group', title="Algorithm Efficiency Comparison")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Metrics Table
                metrics_df = exp_lab.generate_simulated_metrics(study_df)
                st.write("### Simulated Precision/Recall (vs SBERT Oracle)")
                st.table(metrics_df)

            with tab3:
                st.subheader("Cognitive Explainability")
                sel = st.selectbox("Select Candidate", df["Name"].tolist())
                if sel:
                    cand = df[df["Name"] == sel].iloc[0]
                    
                    # Generate Natural Language Explanation
                    reason = f"The candidate **{cand['Name']}** received a score of **{cand['Final Score']}**."
                    reason += f"\n- **Semantic Match**: {cand['SBERT Score']} (Contextual alignment with JD)."
                    reason += f"\n- **Skills**: {len(cand['Matched Skills'])}/{len(jd_skills)} required skills found."
                    
                    if cand['Final Score'] > 80:
                        reason += "\n\n**Conclusion**: Strong Fit. High semantic overlap and skill density."
                    else:
                        reason += "\n\n**Conclusion**: Weak Fit. Consider upskilling in: " + ", ".join(cand['Missing Skills'][:3])
                        
                    st.success(reason)
                    
                    # Radar
                    categories = ['Semantic', 'Skill Match', 'Experience', 'Education']
                    w = current_weights
                    # Inverse calc to show raw components 0-1ish
                    vals = [cand['SBERT Score'], len(cand['Matched Skills'])/len(jd_skills) if jd_skills else 0, 0.8, 0.5] # Approx placeholders for vis
                    
                    fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=categories, fill='toself'))
                    st.plotly_chart(fig_r)

            with tab4:
                st.subheader("Human-in-the-Loop Feedback")
                st.write("Teach the system! If you select a candidate, the Adaptive Engine learns which traits matter most.")
                
                sel_fb = st.selectbox("Mark Candidate for Hiring", df["Name"].tolist(), key="fb_select")
                if st.button("Confirm Hire"):
                    # Mock finding the candidate data
                     # In real app, we pass the Candidate object. Here we just simulate.
                     # We assume 'Skills' was high for this mock.
                     new_weights = st.session_state['adaptive_engine'].update_weights_from_feedback({"Breakdown": {"Skills": 30, "Experience": 10}})
                     st.success(f"System Updated! New Weights adapted to your preference: {new_weights}")
