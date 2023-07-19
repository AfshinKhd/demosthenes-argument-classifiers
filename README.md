# Demosthenes-Argument-Classifiers

In This project aims to develop state-of-the-art argument models for the Demosthenes corpus, a specialized dataset focused on argument mining.Demosthenes is a novel corpus for argument mining in legal documents composed of 40 decisions of the Court of Justice of the European Union on matters of fiscal state aid.

## Corpus GitHub Repository
For access to the Demosthenes corpus, you can visit the GitHub repository at:

[https://github.com/adele-project/demosthenes/tree/main](https://github.com/adele-project/demosthenes/tree/main)

## Dataset Explanation
After executing the xmlToJson.py script, each file contains a JSON file with plain text. The sentences in the JSON file are collecation of sentences as either argumentative or non-argumentative. Additionally, the annotation type indicate the Type and Scheme of each argumentative sentence. By running the create_df.py script, you can obtain two datasets: one that includes all the sentences, and another that reveals premises and conclusions along with their attributes.



