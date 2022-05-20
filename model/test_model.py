import logging
import pandas as pd
import wandb
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pipeline import NumericalTransformer,CategoricalTransformer,FeatureSelector

# global variables

# name of the artifact related to test dataset
artifact_test_name = "Spotify/test.csv:latest"

# name of the model artifact
artifact_model_name = "Spotify/model_export:latest"

# name of the target encoder artifact
artifact_encoder_name = "Spotify/target_encoder:latest"
# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()
# initiate the wandb project
run = wandb.init(project="Spotify",job_type="test")
logger.info("Downloading and reading test artifact")
test_data_path = run.use_artifact(artifact_test_name).file()
df_test = pd.read_csv(test_data_path)

# Extract the target from the features
logger.info("Extracting target from dataframe")
x_test = df_test.copy()
y_test = x_test.pop("genre")

x_test['key'] = x_test['key'].astype('str')
x_test['mode'] = x_test['mode'].astype('str')
x_test['time_signature'] = x_test['time_signature'].astype('str')

# Extract the encoding of the target variable
logger.info("Extracting the encoding of the target variable")
encoder_export_path = run.use_artifact(artifact_encoder_name).file()
le = joblib.load(encoder_export_path)
# transform y_train
y_test = le.transform(y_test)
logger.info("Classes [0 ~ 14]: {}".format(le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])))

logger.info("Downloading and load the exported model")
model_export_path = run.use_artifact(artifact_model_name).file()
pipe = joblib.load(model_export_path)

# predict
logger.info("Infering")
predict = pipe.predict(x_test)

# Evaluation Metrics
logger.info("Test Evaluation metrics")

fbeta = fbeta_score(y_test, predict, beta=1, zero_division=1,average='micro')
precision = precision_score(y_test, predict, zero_division=1,average='micro')
recall = recall_score(y_test, predict, zero_division=1,average='micro')
acc = accuracy_score(y_test, predict)

logger.info("Test Accuracy: {}".format(acc))
logger.info("Test Precision: {}".format(precision))
logger.info("Test Recall: {}".format(recall))
logger.info("Test F1: {}".format(fbeta))

run.summary["Acc"] = acc
run.summary["Precision"] = precision
run.summary["Recall"] = recall
run.summary["F1"] = fbeta

print(classification_report(y_test,predict))

fig_confusion_matrix, ax = plt.subplots(1,1,figsize=(20,20))
ConfusionMatrixDisplay(confusion_matrix(predict,y_test,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]),
                       display_labels=['Dark Trap', 'Underground Rap', 'Trap Metal', 'Emo', 'Rap', 'RnB',
       'Pop', 'Hiphop', 'techhouse', 'techno', 'trance', 'psytrance',
       'trap', 'dnb', 'hardstyle']).plot(values_format=".0f",ax=ax)

ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
plt.show()

# Uploading figures
logger.info("Uploading figures")
run.log(
    {
        "confusion_matrix": wandb.Image(fig_confusion_matrix),
        # "other_figure": wandb.Image(other_fig)
    }
)

run.finish()