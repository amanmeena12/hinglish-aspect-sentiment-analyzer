\# Hinglish Review Aspect Sentiment Analyzer



A Python-based NLP project that analyzes customer reviews written in Hinglish (Hindi + English) and identifies sentiment at both the overall-review and aspect levels.



\## Project Overview



Customer reviews often contain mixed opinions. For example, a customer may like a product's quality but complain about late delivery. This project aims to identify the overall sentiment and analyze individual aspects mentioned in a review.



\## Features



\- Overall review sentiment classification: Positive, Negative, or Neutral

\- Aspect detection for topics such as quality, delivery, price, battery, camera, packaging, customer service, and usability

\- Aspect-level sentiment analysis with supporting review text

\- Batch analysis of multiple reviews using a CSV upload

\- Interactive Streamlit dashboard

\- Sentiment charts and aspect insights

\- Model performance metrics and error analysis

\- CSV export of batch-analysis results



\## Technologies Used



\- Python

\- Pandas

\- Scikit-learn

\- TF-IDF text features

\- Logistic Regression

\- Streamlit

\- Plotly

\- Joblib



\## Project Files



\- `app.py` — Streamlit dashboard

\- `aspect\_analyzer.py` — aspect detection and aspect-level sentiment logic

\- `train\_model.py` — model training and evaluation

\- `create\_dataset.py` — dataset creation utility

\- `hinglish\_reviews.csv` — review dataset

\- `sentiment\_model.joblib` — saved trained model and associated metadata, if included by the training script

\- `error\_analysis.csv` — model error-analysis output



\## Installation



1\. Install Python 3.12 or a compatible Python version.

2\. Open a terminal in the project directory.

3\. Install the required libraries:



&#x20;  ```bash

&#x20;  python -m pip install streamlit pandas scikit-learn plotly joblib

&#x20;  ```



\## Run the Application



Start the dashboard from the project directory:



```bash

python -m streamlit run app.py

```



Open the local URL displayed in the terminal.



\## Train the Model



To retrain the sentiment classifier using the available dataset:



```bash

python train\_model.py

```



Review the terminal output and verify that the model file is generated successfully.



\## How to Use



1\. Open the Streamlit dashboard.

2\. Use \*\*Analyze Review\*\* to analyze a single Hinglish review.

3\. Use \*\*Batch Analysis\*\* to upload a CSV containing a `review` column.

4\. Explore \*\*Aspect Insights\*\* to inspect the aspect and sentiment labels available in the dataset.

5\. Review \*\*Model Performance\*\* and \*\*Error Analysis\*\* to understand evaluation results and classification mistakes.



\## Important Limitations



Aspect detection currently relies on predefined keywords, and aspect-level sentiment uses rule-based logic. The overall sentiment is predicted by a trained machine-learning model. These methods can make mistakes with sarcasm, context, mixed opinions, spelling variations, and complex Hinglish expressions.



Model performance depends on the quality, size, and representativeness of the training data. Dashboard insights based on the existing dataset reflect its stored labels; they do not automatically represent every newly uploaded review.



\## Future Improvements



\- Expand the Hinglish review dataset

\- Improve handling of negation, sarcasm, and mixed sentiments

\- Evaluate aspect-level accuracy on manually labeled test examples

\- Add more product aspects and spelling variations

\- Improve model evaluation and error analysis



\## Author



\*\*Aman Meena\*\*



This project was developed as a portfolio project to practice NLP, machine learning, data analysis, and interactive dashboard development.



