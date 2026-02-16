import os
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.adaptive_engine import RLRankingAgent
from src.experiment import ExperimentLab
from src.matcher import calculate_similarity, calculate_sbert_similarity, find_missing_skills
from src.parser import extract_text
from src.preprocessing import clean_text, extract_skills, preprocess_text, COMMON_SKILLS
from src.research_lab import (
    analyze_fairness,
    calculate_statistics,
    compare_algorithms,
    detect_bias_entities,
    generate_pareto_frontier,
    simulate_rl_convergence,
)
from src.scoring import calculate_readability, calculate_composite_score

# ---------------------------------------------------------------------------
# Session & Page Setup
# ---------------------------------------------------------------------------

# Persist the RL agent across Streamlit reruns
if "adaptive_engine" not in st.session_state:
    st.session_state["adaptive_engine"] = RLRankingAgent()

st.set_page_config(
    page_title="Intelligent AI Resume Screening System",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Intelligent AI Resume Screening System")
st.markdown("### 🚀 Novel Intelligent System with SBERT & Reinforcement Learning")

# ---------------------------------------------------------------------------
# Sidebar — File Upload & Experiment Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF/DOCX files", type=["pdf", "docx"], accept_multiple_files=True
    )
    st.info(f"Loaded: {len(uploaded_files)} resumes")

    st.divider()
    st.subheader("⚙️ Experiment Controls")
    bias_filter = st.checkbox("Enable Bias-Aware Filtering", value=True)
    use_sbert = st.checkbox("Use Deep Semantic (SBERT)", value=True)

    st.markdown("---")
    st.subheader("⚖️ Multi-Objective Balance")
    alpha_input = st.slider(
        "Fairness vs Accuracy Preference (Alpha)",
        min_value=0.0, max_value=1.0, value=0.9, step=0.1,
        help="0.0 = Prioritize Fairness (Blind), 1.0 = Prioritize Accuracy (Raw Match)",
    )

# ---------------------------------------------------------------------------
# Main Content — Job Description Input
# ---------------------------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Job Description")
    jd_input = st.text_area("Paste the Job Description here...", height=300)

# ---------------------------------------------------------------------------
# Run Adaptive Experiment
# ---------------------------------------------------------------------------

if st.button("Run Adaptive Experiment"):
    if not uploaded_files or not jd_input:
        st.warning("Please upload resumes and provide a job description.")
    else:
        with st.spinner("Initializing Deep Neural Networks..."):

            # --- JD Preprocessing ---
            jd_text = clean_text(jd_input)
            jd_processed = preprocess_text(jd_text)
            jd_skills = extract_skills(jd_input)

            # --- RL Agent: detect role context and choose scoring weights ---
            rl_agent = st.session_state["adaptive_engine"]
            role = rl_agent.detect_role(jd_text)
            current_weights = rl_agent.get_weights(explore=True)

            st.toast(f"RL Agent (Role: {role}): Active Weights {current_weights}", icon="🧠")

            # --- Resume Processing ---
            results = []
            cleaned_texts = []

            for file in uploaded_files:
                # Write uploaded bytes to a temp file so parsers can read it
                file_ext = file.name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name

                text = extract_text(tmp_path)
                os.remove(tmp_path)

                if not text:
                    results.append({"Name": file.name, "Processing Status": "Failed"})
                    continue

                # Bias detection (if enabled)
                detected_bias = detect_bias_entities(text) if bias_filter else []
                bias_penalty = 1.0 if detected_bias else 0.0

                # Clean and tokenise
                cleaned = clean_text(text)
                processed = preprocess_text(cleaned)
                cleaned_texts.append(processed)

                results.append({
                    "Name": file.name,
                    "Raw Text": text,
                    "Cleaned Text": processed,
                    "Skills": extract_skills(text),
                    "Detected Bias": detected_bias,
                    "Bias Penalty": bias_penalty,
                    "Readability": calculate_readability(text),
                    "Processing Status": "Success",
                })

            # --- Score Computation ---
            valid_indices = [
                i for i, r in enumerate(results) if r["Processing Status"] == "Success"
            ]

            if valid_indices:
                valid_cleaned = [results[i]["Cleaned Text"] for i in valid_indices]

                # TF-IDF similarity
                tfidf_scores = calculate_similarity(valid_cleaned, jd_processed)

                # SBERT deep semantic similarity (falls back to zeros if disabled)
                sbert_scores = (
                    calculate_sbert_similarity(valid_cleaned, jd_processed)
                    if use_sbert
                    else [0] * len(valid_indices)
                )

                # Attach final composite scores to each result
                for idx, tfidf_score, sbert_score in zip(valid_indices, tfidf_scores, sbert_scores):
                    res = results[idx]

                    # Figure out which skills from the JD this resume matched
                    matched_skills = list(set(res["Skills"]).intersection(set(jd_skills)))

                    # Choose semantic score source
                    semantic_score = sbert_score if use_sbert else tfidf_score

                    final_score, breakdown = calculate_composite_score(
                        resume_text=res["Raw Text"],
                        jd_text=jd_input,
                        skill_match_count=len(matched_skills),
                        total_jd_skills=len(jd_skills),
                        nlp_similarity=tfidf_score,
                        sbert_similarity=semantic_score,
                        weights=current_weights,
                        bias_penalty=res["Bias Penalty"],
                        alpha=alpha_input,
                    )

                    res.update({
                        "NLP Score": round(tfidf_score, 4),
                        "SBERT Score": round(sbert_score, 4),
                        "Final Score": final_score,
                        "RJAS": final_score,  # alias used in visualisations
                        "Matched Skills": matched_skills,
                        "Missing Skills": find_missing_skills(res["Raw Text"], jd_input, COMMON_SKILLS),
                        "Breakdown": breakdown,
                    })

            # ---------------------------------------------------------------
            # Visualisation Tabs
            # ---------------------------------------------------------------

            tab1, tab2, tab3, tab4 = st.tabs([
                "🧠 Intelligent Ranking (RJAS)",
                "🧪 Research Lab",
                "📢 Explainability",
                "🔄 RL Feedback Loop",
            ])

            df = pd.DataFrame([r for r in results if r.get("Processing Status") == "Success"])

            # --- Tab 1: RJAS Ranking ---
            with tab1:
                st.subheader(f"Adaptive Ranking (Role: {role}) | Metric: RJAS")
                if not df.empty:
                    display_cols = ["Name", "RJAS", "SBERT Score", "NLP Score"]
                    df_ranked = df[display_cols].sort_values(by="RJAS", ascending=False)
                    st.dataframe(df_ranked.style.highlight_max(axis=0), use_container_width=True)

                    fig = px.bar(df_ranked, x="Name", y="RJAS", color="RJAS",
                                 title="Resume-Job Alignment Score (RJAS) Ranking")
                    st.plotly_chart(fig, use_container_width=True)

            # --- Tab 2: Research Lab ---
            with tab2:
                st.subheader("Comparative Study & Fairness Analysis")
                st.info("Comparing: Traditional Keyword vs TF-IDF vs Novel RJAS.")

                # Algorithm comparison table
                comparison_data = []
                for res in results:
                    if res["Processing Status"] == "Success":
                        comp = compare_algorithms(res["Raw Text"], jd_input, res["NLP Score"], res["RJAS"])
                        comp["Name"] = res["Name"]
                        comparison_data.append(comp)

                study_df = pd.DataFrame(comparison_data)

                if not study_df.empty:
                    # Grouped bar chart — one cluster per candidate
                    melted = study_df.melt(id_vars=["Name"], var_name="Algorithm", value_name="Score")
                    fig2 = px.bar(melted, x="Name", y="Score", color="Algorithm",
                                  barmode="group", title="Algorithm Efficiency Comparison")
                    st.plotly_chart(fig2, use_container_width=True)

                    # Pareto frontier scatter
                    st.divider()
                    st.subheader("📈 Pareto Frontier Analysis (Accuracy vs Fairness)")
                    pareto_df = generate_pareto_frontier(results, current_weights)

                    if not pareto_df.empty:
                        fig_pareto = px.scatter(
                            pareto_df,
                            x="Global Fairness", y="Global Accuracy",
                            color="Alpha (Preference)",
                            hover_data=["Msg"],
                            title="Pareto Frontier: Trade-off Analysis",
                        )
                        st.plotly_chart(fig_pareto, use_container_width=True)

                # Statistical validation
                st.divider()
                st.subheader("📊 Statistical Validation")
                stats_result = calculate_statistics(df)

                if stats_result:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("RJAS Mean (μ)",
                              f"{stats_result.get('RJAS Mean', 0):.2f}",
                              f"σ={stats_result.get('RJAS Std', 0):.2f}")
                    s2.metric("NLP Mean (μ)",
                              f"{stats_result.get('NLP Mean', 0):.2f}",
                              f"σ={stats_result.get('NLP Std', 0):.2f}")

                    t_val = stats_result.get("T-Statistic")
                    p_val = stats_result.get("P-Value")

                    if t_val is not None:
                        s3.metric("Paired T-Statistic", f"{t_val:.4f}")
                        sig_icon = "✅" if stats_result.get("Significant") else "❌"
                        s4.metric("P-Value (Sig < 0.05)", f"{p_val:.4e} {sig_icon}")
                    else:
                        st.error(f"Stat Error: {stats_result.get('Error')}")

                # Fairness analysis
                st.divider()
                st.subheader("⚖️ Bias-Aware Fairness Constraints")
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

            # --- Tab 3: Explainability ---
            with tab3:
                st.subheader("Cognitive Explainability")
                selected_candidate = st.selectbox("Select Candidate", df["Name"].tolist())

                if selected_candidate:
                    cand = df[df["Name"] == selected_candidate].iloc[0]

                    # Build a natural-language explanation
                    explanation = f"The candidate **{cand['Name']}** achieved an RJAS of **{cand['RJAS']}**."
                    explanation += f"\n- **Semantic Match**: {cand['SBERT Score']} (Contextual alignment)."

                    if cand["Detected Bias"]:
                        explanation += (
                            f"\n- **Fairness**: Detected {len(cand['Detected Bias'])} "
                            "potential bias entities. Penalty applied."
                        )

                    if cand["RJAS"] > 80:
                        explanation += "\n\n**Conclusion**: Superior Fit. Strongly aligned with role requirements."
                    else:
                        explanation += "\n\n**Conclusion**: Moderate/Low Fit."

                    st.success(explanation)

                    # Radar chart of the score breakdown
                    categories = ["Semantic", "Skill", "Experience", "Education"]
                    breakdown_scores = cand["Breakdown"]
                    radar_values = [
                        breakdown_scores.get("Semantic", 0) / 100,
                        breakdown_scores["Skills"] / 100,
                        breakdown_scores["Experience"] / 100,
                        breakdown_scores["Education"] / 100,
                    ]

                    fig_radar = go.Figure(
                        data=go.Scatterpolar(r=radar_values, theta=categories, fill="toself")
                    )
                    st.plotly_chart(fig_radar)

            # --- Tab 4: RL Feedback Loop ---
            with tab4:
                st.subheader("Human-in-the-Loop RL Feedback")
                st.write("Teach the system! The RL Agent updates its policy based on your hiring decisions.")

                selected_hire = st.selectbox("Mark Candidate for Hiring", df["Name"].tolist(), key="fb_select")

                if st.button("Confirm Hire"):
                    hired_data = df[df["Name"] == selected_hire].iloc[0]
                    new_weights = st.session_state["adaptive_engine"].update_policy(hired_data, reward=1.0)
                    st.balloons()
                    st.success(f"RL Agent Updated! New Weights for '{role}': {new_weights}")

                st.divider()
                st.subheader("🔬 RL Convergence Simulator (Research Mode)")

                if st.button("Run Simulation (100 Iterations)"):
                    with st.spinner("Simulating user feedback loop..."):
                        sim_data = simulate_rl_convergence(iterations=100, role=role)
                        st.success("Simulation Complete!")

                        # Plot weight evolution over time
                        weight_cols = [c for c in sim_data.columns if c not in ["Iteration", "Cumulative Reward"]]
                        st.line_chart(sim_data, x="Iteration", y=weight_cols)
                        st.caption(
                            "Weight Evolution: Observation of how the agent adapts to the "
                            "'Ideal' candidate profile over time."
                        )
