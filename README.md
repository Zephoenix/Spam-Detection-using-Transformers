# Spam Detection Using Transformers

This project addresses the persistent challenge of distinguishing between spam and legitimate (ham) emails using both traditional machine learning techniques and a fine-tuned BERT-based transformer model.

## Project Overview

With over 50% of global email traffic classified as spam, designing a reliable spam detection system is critical for personal and enterprise communication. This project aims to:
- Minimize false negatives (ensure spam isn't missed).
- Adapt to evolving spam strategies.
- Benchmark classical ML models against state-of-the-art deep learning techniques.

## Dataset

We utilized a hybrid dataset consisting of:
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/): 33,716 emails (17,171 spam, 16,545 ham).
- Manually labeled personal email samples.
- Synthetic adversarial samples generated using OpenAI’s GPT-4 for robustness testing.

## Methodology

### 1. Baseline Models
- Logistic Regression
- Multinomial Naïve Bayes
- Support Vector Classifier (SVC)

### 2. Advanced Model
- Fine-tuned [BERT (Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805) via Hugging Face Transformers.

### 3. Feature Engineering
- Count-based: Word counts, stop words, punctuation.
- Sentiment-based: AFINN, Bing Liu, NRC Lexicons.
- Frequency-based: TF-IDF vectors.
- Bag-of-Words: Basic vector representations.

## Results

| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Naïve Bayes         | 91.8%    | 93.3%     | 90.7%  | 92.0%    | —       |
| SVC                 | 93.3%    | 92.4%     | 94.9%  | 93.6%    | —       |
| BERT (Enron)        | 98.8%    | 98.8%     | 99.0%  | 98.9%    | 100.0%  |
| BERT (Adversarial)  | 99.5%    | 99.3%     | 100.0% | 99.6%    | —       |

## Running the Code

```bash
# 1. Clone the repository
git clone https://github.com/Zephoenix/Spam-Detection-using-Transformers.git
cd Spam-Detection-using-Transformers

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the demo
python "demo files/Project_demo.py"
```

## File Structure

- `ConvertEmails.py` – Preprocesses email files.
- `Final_project _code.py` – Full model training and evaluation.
- `LLMEmailGeneration.py` – Uses ChatGPT to create adversarial examples.
- `model results/` – Includes pretrained tokenizer and BERT model configs.
- `demo files/` – Demo app with web UI.
- `PersonalEmailSampling.Rmd` – Analysis of personal email data in R.

## References

- Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.”
- [Statista](https://www.statista.com) on global email spam traffic.
- OpenAI GPT-4 for adversarial email generation.
