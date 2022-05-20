import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import tempfile
import os

input_artifact="Spotify/raw_data.csv:latest"
artifact_name="preprocessed_data.csv"
artifact_type="clean_data"
artifact_description="Data after preprocessing"

run = wandb.init(project="Spotify", job_type="process_data")

# donwload the latest version of artifact raw_data.csv
artifact = run.use_artifact(input_artifact)

# create a dataframe from the artifact
df = pd.read_csv(artifact.file(),sep=',',low_memory=False)

#check dataframe
df.head()

# Delete some that will not by used on the training
df = df.drop(columns = ['type', 'id', 'uri', 'track_href', 'analysis_url','song_name', 'Unnamed: 0', 'title'])

# Delete duplicated rows
df = df.drop_duplicates()

# Generate a "csv for the preprocessed data file"
df.to_csv('preprocessed_data.csv',index=False)
df = pd.read_csv('preprocessed_data.csv')

#do a exploratory data analysis(EDA) of the preprocessed dataframe
ProfileReport(df, title="Pandas Profiling Report", explorative=True)

# Create a new artifact and configure with the necessary arguments
artifact = wandb.Artifact(name=artifact_name,
                          type=artifact_type,
                          description=artifact_description)
artifact.add_file(artifact_name)

# Upload the artifact to Wandb
run.log_artifact(artifact)

run.finish()