# SkillBridge AI — ML Training Notebooks

**AI-Powered Career Guidance and Job Recommendation System**
Aligned with UN Sustainable Development Goal 8 (Decent Work and Economic Growth)

---

## Overview

SkillBridge AI is an end-to-end machine learning system that helps students and fresh graduates discover suitable career paths, receive personalized job recommendations, and get AI-driven feedback on their CVs. The system supports CV uploads in PDF, DOCX, and image (OCR) formats and produces ranked job recommendations through a multi-model ensemble pipeline.

This repository contains the full history of Google Colab training notebooks, from the initial build through the final production-ready version. Each version introduced new models, research paper implementations, and evaluation techniques.

---

## Repository Structure

```
SkillBridge-AI-ML/
│
├── notebooks/
│   ├── SkillBridge_AI_ML_Training_v2.ipynb
│   ├── SkillBridge_AI_ML_Training_v3.ipynb
│   ├── SkillBridge_AI_ML_Training_v3_1_.ipynb
│   ├── SkillBridge_AI_ML_Training_v4.ipynb
│   ├── SkillBridge_AI_ML_Training_v4_1_.ipynb
│   ├── SkillBridge_AI_ML_Training_v5.ipynb
│   ├── SkillBridge_AI_ML_Training_v6.ipynb       ← Final version
│   ├── SkillBridge_AI_ML_Fixed.ipynb
│   ├── SkillBridge_AI_ML_Training_Demo.ipynb
│   ├── ML_Project_Demo.ipynb
│   └── ML_Demo_Project.ipynb
│
└── README.md
```

---

## Notebook Version History

### v2 — Foundation Build
The first structured version of the training pipeline. Implements the core research paper methods and establishes the baseline recommendation system.

- TF-IDF vectorization with cosine similarity (Ajjam & Al-Raweshidy, 2026)
- Weighted cosine similarity and Jaccard coefficient (Alsaif et al., 2022)
- Word2Vec semantic embeddings
- CV upload support for PDF and DOCX formats
- Basic Flask API endpoint structure

---

### v3 / v3.1 — Expanded Research Integration
Targets young Bangladeshi students and fresh graduates specifically. Extends the research implementation and improves the recommendation quality.

- Personalized education and skill-to-job alignment (Tavakoli et al., 2022)
- GRU-based career tracking with attention mechanism (Huang, 2022)
- 6-stage recruitment pipeline with bias detection (Zhisheng Chen, 2022)
- Improved data preprocessing and EDA
- Extended evaluation metrics

---

### v4 / v4.1 — Pipeline Consolidation
Consolidates all prior models into a cleaner, more stable pipeline with improved structure and reliability.

- Refined ensemble recommendation combining all five models
- Improved CV parser with multi-format support (PDF, DOCX, OCR image)
- VADER sentiment analysis applied to CV text
- P-J Fit and P-O Fit scoring for resume evaluation
- Personalized learning pathway recommendations
- Salary prediction using linear regression on CV features

---

### v5 — Job Title Classifier Added
Introduces a dedicated job title classification model trained on a large real-world dataset.

- Job Title Classifier: RandomForestClassifier + TF-IDF
- Trained on 50,000 rows with 639 distinct job title classes
- New Flask endpoint: `/job_title_predict`
- Integrated into the recommendation pipeline

---

### v6 — Final Production Version (100 Cells)
The most complete and research-rigorous version of the notebook. This is the primary deliverable for the SkillBridge AI system.

**Full section list:**

| Section | Description |
|---------|-------------|
| 1 | Mount Drive and install all dependencies |
| 2 | Full imports (scikit-learn, gensim, nltk, pdfplumber, etc.) |
| 3 | Global configuration |
| 4 | Load dataset from Kaggle |
| 5 | EDA — correlation analysis and distributions |
| 6 | Data preprocessing and cleaning |
| 7 | Custom TF-IDF feature engineering |
| 8 | Word2Vec semantic embeddings |
| 9 | Jaccard coefficient skill similarity |
| 10 | Core model — TF-IDF + cosine similarity |
| 11 | KNN recommender using NearestNeighbors |
| 12 | GRU career tracker with attention and greedy algorithm |
| 12.5 | Skills space mapping and career transition pathways |
| 13 | 5-model ensemble recommendation pipeline |
| 14 | CV parser — PDF, DOCX, OCR image |
| 14.1 | CV sentiment enrichment using VADER |
| 14.5 | Personalized learning pathway recommendations |
| 14.6 | P-J Fit, P-O Fit, and AI resume scoring |
| 15 | Linear regression — GPA to job success prediction |
| 16 | Decision Tree and Random Forest classifiers |
| 17 | KNN classifier for career path prediction |
| 18 | Salary prediction from CV features |
| 19 | Complete evaluation — Precision@K, Recall@K, F1, AUC, Wilcoxon, Cohen's D |
| 19.5 | AUC-ROC evaluation |
| 19.6 | Employment confidence scoring and regression analysis |
| 19.7 | IPW robustness test — causal effect of AI tool use |
| 19.8 | Heterogeneity analysis by academic level and field |
| 19.9 | SEM mediation pathway analysis |
| 20.5 | LDA topic modelling for labour market parameters |
| 20.6 | Similarity-based negative sampling module |
| 20.7 | Employment intention classifier — 3 classes |
| 20.8 | AUC ablation study — method comparison |
| 20.9 | Job Title Classifier — RandomForest + TF-IDF (639 classes) |
| 21.1 | Comprehensive ML data preparation and feature engineering |
| 21.2 | 7-model classification benchmark with runtime and accuracy |
| 21.3 | Stratified cross-validation with 95% confidence intervals |
| 21.4 | Polynomial regression with 95% prediction interval |
| 21.5 | Statistical significance tests — ANOVA and Wilcoxon |
| 21.6 | Binary genetic algorithm for feature selection |
| 21.7 | Deep ANN — architecture search and analysis |
| 21.8 | Comprehensive performance metrics dashboard |
| 21.9 | Comprehensive visualizations dashboard |
| 21.10 | Comparative data analysis and interpretation |
| 20 | Export all model artifacts as .pkl files for Flask API |
| 21 | Complete Flask API with CV upload endpoint |
| 22 | Final summary report |

---

### Fixed — Bug-Fixed Notebook
A patched version resolving cell-indexing issues and string-escaping errors found during Flask API integration testing.

---

### Demo Notebooks
Three demonstration notebooks designed for presentation and academic review:

- **SkillBridge_AI_ML_Training_Demo.ipynb** — Streamlined walkthrough of the full pipeline for live demonstrations
- **ML_Project_Demo.ipynb** — AI Career Guidance System v12.0, research proposal edition
- **ML_Demo_Project.ipynb** — Alternative demo configuration

---

## Research Papers Implemented

| Author | Year | Method |
|--------|------|--------|
| Ajjam & Al-Raweshidy | 2026 | TF-IDF + Cosine Similarity + Greedy Matching |
| Alsaif et al. | 2022 | Weighted Cosine Similarity + Jaccard + Word2Vec |
| Tavakoli et al. | 2022 | eDoer Personalized Education (Skill-to-Job Alignment) |
| Huang | 2022 | GRU Career Tracking + Attention Mechanism |
| Zhisheng Chen | 2022 | 6-Stage Recruitment Pipeline + Bias Detection |
| Alaql et al. | — | Feature selection and classification benchmarking |
| Dawson et al. | — | Statistical significance in educational ML models |
| Xiao & Zheng | — | Semantic similarity in job recommendation |

---

## Models and Techniques

- TF-IDF Vectorization and Cosine Similarity
- Jaccard Coefficient for skill matching
- Word2Vec for semantic skill embeddings
- GRU (Gated Recurrent Unit) with attention mechanism
- K-Nearest Neighbors (KNN) for career path prediction
- Random Forest Classifier (job title prediction, 639 classes)
- Decision Tree Classifier
- Linear Regression (GPA-to-success, salary prediction)
- Polynomial Regression
- Latent Dirichlet Allocation (LDA) topic modelling
- VADER Sentiment Analysis
- BM25 ranking
- SBERT semantic re-ranking
- Binary Genetic Algorithm for feature selection
- Deep Artificial Neural Network (ANN)
- 5-model ensemble recommendation
- 7-model classification benchmark

---

## Evaluation Metrics

- Precision@K and Recall@K
- F1 Score and AUC-ROC
- Wilcoxon Signed-Rank Test
- Cohen's D effect size
- One-Way ANOVA
- 95% Bootstrap Confidence Intervals
- Stratified Cross-Validation
- Ablation Study (method comparison)
- IPW Robustness Test
- SEM Mediation Analysis

---

## Flask API Endpoints

The notebooks export trained model artifacts (`.pkl` files) for use in the Flask backend.

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/recommend` | Job recommendations from skill input |
| `/cv_recommend` | Job recommendations from uploaded CV |
| `/career_predict` | Career path prediction |
| `/job_title_predict` | Job title classification (639 classes) |

---

## How to Run

1. Open any notebook in [Google Colab](https://colab.research.google.com)
2. Upload the file or open directly from Google Drive
3. Run Section 1 first to mount your Drive and install all dependencies
4. Place the dataset file `job_recommendation_dataset.csv` in your Google Drive
5. Run all cells from top to bottom
6. Model `.pkl` files will be saved to your Drive for use with the Flask API

**Recommended starting point:** `SkillBridge_AI_ML_Training_v6.ipynb`

---

## Dataset

The system uses `job_recommendation_dataset.csv` stored on Google Drive. The dataset contains job titles, required skills, industries, and candidate profiles used to train all recommendation and classification models.

- Job Title Classifier training set: 50,000 rows, 639 classes

---

## System Architecture

```
Flutter Android App
        |
        v
  Flask REST API  (Render.com)
        |
        v
  ML Models (.pkl)
  - TF-IDF Vectorizer
  - Cosine Similarity Engine
  - KNN Recommender
  - GRU Career Tracker
  - Random Forest Job Title Classifier
  - Ensemble Pipeline
        |
        v
  Google Colab Training Notebooks (this repository)
```

---

## SDG Alignment

This project contributes to **UN SDG 8: Decent Work and Economic Growth** by providing AI-powered career guidance to students and fresh graduates, reducing skill-job mismatch, and improving employment outcomes for young people in developing regions.

---

## License

This project is developed for academic and research purposes.

---

## Author

Developed as part of an AI-powered career guidance research project targeting SDG-8 outcomes.
