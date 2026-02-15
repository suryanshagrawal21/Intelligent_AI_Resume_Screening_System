# Intelligent AI Resume Screening System

A sophisticated resume screening system powered by SBERT (Sentence-BERT) and Reinforcement Learning to intelligently match candidates to job descriptions.

## 🚀 Features

- **Intelligent Matching**: Uses SBERT for deep semantic understanding of resumes and job descriptions.
- **Adaptive Engine**: Learns from user feedback to adjust weighting for skills, experience, and education.
- **Bias-Aware Filtering**: Automatically detects and can filter potential bias entities.
- **Explainability**: Provides natural language explanations for candidate scores.
- **Comparative Analysis**: Compares different algorithms (TF-IDF, SBERT, Custom) to show performance metrics.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/suryanshagrawal21/NLP-Resume-Screening-System.git
    cd Intelligent_AI_Resume_Screening_System
    ```

2.  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Download necessary NLTK data (if prompted or add to a startup script):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## 🏃 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to the URL provided (usually `http://localhost:8501`).

## 📂 Project Structure

- `app.py`: Main Streamlit application.
- `src/`: Source code for processing, matching, and scoring.
- `data/`: Directory for storing data (if any).
- `notebooks/`: Jupyter notebooks for experiments.
- `docs/`: Documentation files.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
