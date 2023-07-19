import pandas as pd

# read data
df_sentences = pd.read_pickle("..\df_sentences.pkl")
df_annotations = pd.read_pickle("..\df_annotations.pkl")

print(df_annotations)