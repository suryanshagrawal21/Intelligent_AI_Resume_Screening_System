import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.parser import extract_text
from src.preprocessing import clean_text, extract_skills, preprocess_text, COMMON_SKILLS
from src.matcher import calculate_similarity, calculate_sbert_similarity, find_missing_skills
from src.research_lab import compare_algorithms, detect_bias_entities
from src.scoring import calculate_readability
from src.adaptive_engine import RLRankingAgent
from src.experiment import ExperimentLab

# Initialize Engines
if 'adaptive_engine' not in st.session_state:
    st.session_state['adaptive_engine'] = RLRankingAgent()

# Page Config
st.set_page_config(page_title="Intelligent AI Resume Screening System", page_icon="🧬", layout="wide")

st.title("🧬 Intelligent AI Resume Screening System")
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
    
    st.markdown("---")
    st.subheader("⚖️ Multi-Objective Balance")
    alpha_input = st.slider("Fairness vs Accuracy Preference (Alpha)", 0.0, 1.0, 0.9, 0.1, help="0.0 = Prioritize Fairness (Blind), 1.0 = Prioritize Accuracy (Raw Match)")

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
            
            # RL AGENT: Get weights (with exploration)
            current_weights = st.session_state['adaptive_engine'].get_weights(explore=True)
            
            st.toast(f"RL Agent (Role: {role}): Active Weights {current_weights}", icon="🧠")
            
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
                    bias_penalty = 1.0 if detected_bias else 0.0
                    
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
                        "Bias Penalty": bias_penalty,
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
                
                from src.scoring import calculate_composite_score
                
                for idx, (t_score, s_score) in zip(valid_indices, zip(tfidf_scores, sbert_scores)):
                    res = results[idx]
                    
                    # Calculate RJAS
                    skill_match = list(set(res["Skills"]).intersection(set(jd_skills)))
                    
                    # Use SBERT/TF-IDF as semantic score
                    semantic_score = s_score if use_sbert else t_score
                    
                    final_score, breakdown = calculate_composite_score(
                        resume_text=res["Raw Text"],
                        jd_text=jd_input,
                        skill_match_count=len(skill_match),
                        total_jd_skills=len(jd_skills),
                        nlp_similarity=t_score,
                        sbert_similarity=semantic_score,
                        weights=current_weights,
                        bias_penalty=res["Bias Penalty"],
                        alpha=alpha_input
                    )
                    
                    res.update({
                        "NLP Score": round(t_score, 4),
                        "SBERT Score": round(s_score, 4),
                        "Final Score": final_score,
                        "RJAS": final_score,     # Alias for clarity
                        "Matched Skills": skill_match,
                        "Missing Skills": find_missing_skills(res["Raw Text"], jd_input, COMMON_SKILLS),
                        "Breakdown": breakdown
                    })

            # --- VISUALIZATION TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["🧠 Intelligent Ranking (RJAS)", "🧪 Research Lab", "📢 Explainability", "🔄 RL Feedback Loop"])
            
            df = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])
            
            with tab1:
                st.subheader(f"Adaptive Ranking (Role: {role}) | Metric: RJAS")
                if not df.empty:
                    df_display = df[["Name", "RJAS", "SBERT Score", "NLP Score"]].sort_values(by="RJAS", ascending=False)
                    st.dataframe(df_display.style.highlight_max(axis=0), use_container_width=True)
                    
                    fig = px.bar(df_display, x="Name", y="RJAS", color="RJAS", title="Resume-Job Alignment Score (RJAS) Ranking")
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Comparative Study & Fairness Analysis")
                st.info("Comparing: Traditional Keyword vs TF-IDF vs Novel RJAS.")
                
                exp_lab = ExperimentLab()
                # Run the lab analysis
                # Pass RJAS scores to the comparison function inside a wrapper or modify loop
                
                comparison_data = []
                for res in results:
                    if res["Processing Status"] == "Success":
                        comp = compare_algorithms(res["Raw Text"], jd_input, res["NLP Score"], res["RJAS"])
                        comp["Name"] = res["Name"]
                        comparison_data.append(comp)
                
                study_df = pd.DataFrame(comparison_data)
                
                if not study_df.empty:
                     # Melt for chart
                    study_melt = study_df.melt(id_vars=["Name"], var_name="Algorithm", value_name="Score")
                    fig2 = px.bar(study_melt, x="Name", y="Score", color="Algorithm", barmode='group', title="Algorithm Efficiency Comparison")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    st.divider()
                    st.subheader("📈 Pareto Frontier Analysis (Accuracy vs Fairness)")
                    from src.research_lab import generate_pareto_frontier
                    pareto_df = generate_pareto_frontier(results, current_weights)
                    
                    if not pareto_df.empty:
                        fig_p = px.scatter(pareto_df, x="Global Fairness", y="Global Accuracy", color="Alpha (Preference)", 
                                         hover_data=["Msg"], title="Pareto Frontier: Trade-off Analysis")
                        st.plotly_chart(fig_p, use_container_width=True)
                
                st.divider()
                st.subheader("⚖️ Bias-Aware Fairness Constraints")
                from src.research_lab import analyze_fairness
                fairness_metrics = analyze_fairness(df)
                
                if fairness_metrics:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg RJAS (Bias Detected)", fairness_metrics["Avg Score (Bias Detected)"])
                    c2.metric("Avg RJAS (Clean)", fairness_metrics["Avg Score (Clean)"])
                    c3.metric("Fairness Gap", fairness_metrics["Gap"], delta_color="inverse")
                    if fairness_metrics["Gap"] > 10:
                        st.warning("⚠️ Significant Gap Detected! Bias penalties are active.")
                    else:
                        st.success("✅ Ranking appears balanced.")

            with tab3:
                st.subheader("Cognitive Explainability")
                sel = st.selectbox("Select Candidate", df["Name"].tolist())
                if sel:
                    cand = df[df["Name"] == sel].iloc[0]
                    
                    # Generate Natural Language Explanation
                    reason = f"The candidate **{cand['Name']}** achieved an RJAS of **{cand['RJAS']}**."
                    reason += f"\n- **Semantic Match**: {cand['SBERT Score']} (Contextual alignment)."
                    if cand['Detected Bias']:
                         reason += f"\n- **Fairness**: Detected {len(cand['Detected Bias'])} potential bias entities. Penalty applied."
                    
                    if cand['RJAS'] > 80:
                        reason += "\n\n**Conclusion**: Superior Fit. Strongly aligned with role requirements."
                    else:
                        reason += "\n\n**Conclusion**: Moderate/Low Fit."
                        
                    st.success(reason)
                    
                    # Radar
                    categories = ['Semantic', 'Skill', 'Experience', 'Education']
                    # Get normalized values for Radar
                    # breakdown has raw 0-100 or ratio
                    # We need 0-1 for radar
                    bd = cand["Breakdown"]
                    vals = [
                        bd["Semantic"]/100 if "Semantic" in bd else 0,
                        bd["Skills"]/100,
                        bd["Experience"]/100,
                        bd["Education"]/100
                    ]
                    
                    fig_r = go.Figure(data=go.Scatterpolar(r=vals, theta=categories, fill='toself'))
                    st.plotly_chart(fig_r)

            with tab4:
                st.subheader("Human-in-the-Loop RL Feedback")
                st.write("Teach the system! The RL Agent updates its policy based on your hiring decisions.")
                
                sel_fb = st.selectbox("Mark Candidate for Hiring", df["Name"].tolist(), key="fb_select")
                if st.button("Confirm Hire"):
                     # Update RL Agent
                     cand_data = df[df["Name"] == sel_fb].iloc[0]
                     new_weights = st.session_state['adaptive_engine'].update_policy(cand_data, reward=1.0)
                     st.balloons()
                     st.success(f"RL Agent Updated! New Weights for '{role}': {new_weights}")
