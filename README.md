# CreatorLens AI

**Understand · Analyze · Grow**: a multilingual social media intelligence and audience understanding system.
A Streamlit prototype that turns English, Tamil and Tanglish (Romanised Tamil) comments into ranked content ideas, FAQ clusters and a review queue.

## What's new in this version

- **Sunset Silhouette theme**: espresso `#291A17`, copper `#B8713B`, cream `#F7D89E`, slate `#A9ACB3` (light, warm layout)
- **Dashboard redesigned** to the new layout: greeting + date range + export, hero banner, four KPI cards, sentiment & emotion analysis with a 7-day trend, top audience topics, language distribution, content opportunities, recent highlights, audience questions, comment explorer and quick stats
- **Seven pages**: Dashboard, Comment Explorer, Audience Insights, Content Opportunities, Language Intelligence, Moderation & Toxicity, Reports

## Features

- **Module 1:** text normalisation, emoji feature extraction, language tagging (English / Tamil / Code-Mixed)
- **Module 2:** sentiment and intent classifier (heuristic stand-in for XLM-RoBERTa / MuRIL)
- **Module 3:** semantic grouping and FAQ extraction (character n-gram TF-IDF + cosine similarity)
- **Module 4:** opportunity ranker: `Score = (similar requests × 1.5) + (avg likes × 0.8) + structural severity weight`
- Flagged review queue and CSV export

## Project structure

```
creatorlens-ai/
├── app.py                  # entire application (theme CSS, pipeline, pages)
├── requirements.txt
├── .streamlit/config.toml  # light Sunset Silhouette theme
├── .vscode/                # VS Code settings + debug launcher
└── .gitignore
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. In VS Code: open the folder, select the `.venv` interpreter, then press **F5**.

## Customising

- **Colours:** the `C` dict and the `:root` block at the top of `app.py`, plus `.streamlit/config.toml`.
- **Dashboard figures** (2,480 comments, 68% positive, topics, opportunities, highlights…): the `DASHBOARD CONTENT` block in `app.py`.
  The other six pages are computed live from the comment pipeline.
- **Real data:** replace `get_mock_data()` with YouTube / Instagram comment exports, and `classify_comment()` with a multilingual transformer.

## Push to GitHub / Deploy

```bash
git init && git add . && git commit -m "CreatorLens AI: Sunset Silhouette redesign"
git branch -M main
git remote add origin https://github.com/<your-username>/creatorlens-ai.git
git push -u origin main
```

Then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`.
