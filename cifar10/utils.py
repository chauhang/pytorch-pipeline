import pandas as pd
import json

from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(actuals, preds, output_path):
    confusion_matrix_output = confusion_matrix(actuals, preds)
    confusion_df = pd.DataFrame(confusion_matrix_output)
    confusion_df.to_csv(output_path, index=False)
