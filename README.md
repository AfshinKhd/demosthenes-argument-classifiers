# Demosthenes-Argument-Classifiers

In This project aims to develop state-of-the-art argument models for the Demosthenes corpus, a specialized dataset focused on argument mining.Demosthenes is a novel corpus for argument mining in legal documents composed of 40 decisions of the Court of Justice of the European Union on matters of fiscal state aid.

## Corpus GitHub Repository
For access to the Demosthenes corpus, you can visit the GitHub repository at:

[https://github.com/adele-project/demosthenes/tree/main](https://github.com/adele-project/demosthenes/tree/main)

## Dataset Explanation
After executing the xmlToJson.py script, each file contains a JSON file with plain text. The sentences in the JSON file are collecation of sentences as either argumentative or non-argumentative. Additionally, the annotation type indicate the Type and Scheme of each argumentative sentence. By running the create_df.py script, you can obtain two datasets: one that includes all the sentences, and another that reveals premises and conclusions along with their attributes.

## Investigating Corpus
For studying this corpus, 4 tasks have been implemented, consisting of:
1. Argument Detection (AD): given a sentence, classify it as premise, conclusion, or neither.
2. Argument Classification (AC): given a sentence that is known to be argumentative, classify it as premise or conclusion.
3. Type Classification (TC): given a sentence that is known to be a premise is classified as legal(L) and/or factual (F).
4. Scheme Classification (SC): given a sentence, known to be a legal premise, is classified according to its scheme. The Schemes include Rule, Precedent, Authoritative, Classification, Interpretative and Principle.

## Get started
To get started with project, follow these simple steps:

1. **Install Requirements:**

   Make sure you have set up a virtual environment and activate it. Then, install the project dependencies listed in the `requirments.txt` file using the following command:

   ```bash
   pip install -r requirment.txt
2. **Run the Project:**

   To run the project, use the train_net.py script and specify the configuration file using the --config-file flag. For example:
   ```bash
   python train_net.py --config-file distilbert.yaml

## Evaluation Result
In Studying *Argument Mining, following result are achieved:
1. Argument Detection(AD)

| Model       | Metric    | Premise | Conclusion | Neither | micro avg | macro avg |
|-------------|-----------|---------|------------|---------|-----------|-----------|
| Distilbert  | Precision | 0.75    | 0.63       | 0.91    | 0.86      | 0.76      |
| Distilbert  | Recall    | 0.73    | 0.79       | 0.91    | 0.86      | 0.76      |
| Distilbert  | f1-score  | 0.74    | 0.70       | 0.91    | 0.86      | 0.78      |
| Mini-lm     | Precision | 0.70    | 0.70       | 0       |           |           |
| Mini-lm     | Recall    | 0.70    | 0.70       | 0       |           |           |
| Mini-lm     | f1-score  | 0.70    | 0.70       | 0       |           |           |
| xlm-roberta | Precision | 0.70    | 0.70       | 0       |           |           |
| xlm-roberta | Recall    | 0.70    | 0.70       | 0       |           |           |
| xlm-roberta | f1-score  | 0.70    | 0.70       | 0       |           |           |


2. Argument Classification(AC)

| Model       | Metric    | Premise | Conclusion | micro avg | macro avg |
|-------------|-----------|---------|------------|-----------|-----------|
| Distilbert  | Precision | 0.98    | 0.77       | 0.97      | 0.88      |
| Distilbert  | Recall    | 0.99    | 0.65       | 0.97      | 0.82      |
| Distilbert  | f1-score  | 0.99    | 0.71       | 0.97      | 0.85      |
| Mini-lm     | Precision | 0.70    | 0.70       |           |           |
| Mini-lm     | Recall    | 0.70    | 0.70       |           |           |
| Mini-lm     | f1-score  | 0.70    | 0.70       |           |           |
| xlm-roberta | Precision | 0.70    | 0.70       |           |           |
| xlm-roberta | Recall    | 0.70    | 0.70       |           |           |
| xlm-roberta | f1-score  | 0.70    | 0.70       |           |           |


3. Type Classification(TC)

| Model       | Metric    | Fact  | Logic | micro avg | macro avg |
|-------------|-----------|-------|-------|-----------|-----------|
| Distilbert  | Precision | 0.85  | 0.90  | 0.86      | 0.87      |
| Distilbert  | Recall    | 0.94  | 0.75  | 0.86      | 0.84      |
| Distilbert  | f1-score  | 0.89  | 0.81  | 0.86      | 0.85      |
| Mini-lm     | Precision | 0.70  | 0.70  |           |           |
| Mini-lm     | Recall    | 0.70  | 0.70  |           |           |
| Mini-lm     | f1-score  | 0.70  | 0.70  |           |           |
| xlm-roberta | Precision | 0.70  | 0.70  |           |           |
| xlm-roberta | Recall    | 0.70  | 0.70  |           |           |
| xlm-roberta | f1-score  | 0.70  | 0.70  |           |           |



4. Scheme Classification(SC)

| Model       | Metric    | Rule | Itpr | Prec | Class | Princ | Aut  | micro avg | macro avg |
|-------------|-----------|------|------|------|-------|-------|------|-----------|-----------|
| Distilbert  | Precision | 0.57 | 0.87 | 0.67 | 1.00  | 0     | 0.64 | 0.67      | 0.62      |
| Distilbert  | Recall    | 0.54 | 0.41 | 0.90 | 0.33  | 0     | 0.64 | 0.63      | 0.47      |
| Distilbert  | f1-score  | 0.55 | 0.56 | 0.77 | 0.50  | 0     | 0.64 | 0.65      | 0.50      |
| Mini-lm     | Precision | 0    | 0    | 0.70 | 0     | 0     | 0    |           |           |
| Mini-lm     | Recall    | 0    | 0    | 0.70 | 0     | 0     | 0    |           |           |
| Mini-lm     | f1-score  | 0    | 0    | 0.70 | 0     | 0     | 0    |           |           |
| xlm-roberta | Precision | 0    | 0    | 0.70 | 0     | 0     | 0    |           |           |
| xlm-roberta | Recall    | 0    | 0    | 0.70 | 0     | 0     | 0    |           |           |
| xlm-roberta | f1-score  | 0    | 0    | 0.70 | 0     | 0     | 0    |           |           |


