
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('metaflow')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('tensorflow')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('tensorflow_decision_forests')



def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('scikit-learn')

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pylint')

from metaflow import FlowSpec, step, IncludeFile, Parameter
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from io import StringIO
from sklearn.model_selection import train_test_split

class TitanicFlow(FlowSpec):

    DATA_FILE = IncludeFile(
        'dataset',
        help='CSV file with the Titanic dataset',
        is_text=True,
        default='train.csv')

    TEST_FILE = IncludeFile(
        'testset',
        help='CSV file with the Titanic test dataset',
        is_text=True,
        default='test.csv')

    num_trees = Parameter('num_trees',
                          help='Number of trees for the model',
                          default=1000)

    def tokenize_names(self, features, labels=None):
        features["Name"] =  tf.strings.split(features["Name"])
        return features, labels

    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        self.train_df = pd.read_csv(StringIO(self.DATA_FILE))
        self.serving_df = pd.read_csv(StringIO(self.TEST_FILE))
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        def preprocess(df):
            df = df.copy()
            def normalize_name(x):
                return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
            def ticket_number(x):
                return x.split(" ")[-1]
            def ticket_item(x):
                items = x.split(" ")
                if len(items) == 1:
                    return "NONE"
                return "_".join(items[0:-1])
            df["Name"] = df["Name"].apply(normalize_name)
            df["Ticket_number"] = df["Ticket"].apply(ticket_number)
            df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
            return df

        self.preprocessed_train_df = preprocess(self.train_df)
        self.preprocessed_serving_df = preprocess(self.serving_df)
        self.next(self.split_data)

    @step
    def split_data(self):
        self.train_df, self.validation_df = train_test_split(self.preprocessed_train_df, test_size=0.2, random_state=42)
        self.next(self.train_model)

    @step
    def train_model(self):
        input_features = list(self.train_df.columns)
        input_features.remove("Ticket")
        input_features.remove("PassengerId")
        input_features.remove("Survived")
        train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.train_df,label="Survived").map(self.tokenize_names)

        model = tfdf.keras.GradientBoostedTreesModel(
                verbose=0,
                features=[tfdf.keras.FeatureUsage(name=n) for n in input_features],
                exclude_non_specified_features=True,
                num_trees=int(self.num_trees),
                random_seed=1234,
            )
        model.fit(train_ds)

        # Save the model to a file
        model.save('model_%s' % self.num_trees)
        self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        # Load the model from the file
        model = tf.keras.models.load_model('model_%s' % self.num_trees)

        validation_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.validation_df, label="Survived").map(self.tokenize_names)

        predictions = model.predict(validation_ds)
        predictions = (predictions >= 0.5).astype(int)

        labels = self.validation_df["Survived"].values

        self.accuracy = (predictions == labels).mean()
        self.next(self.make_predictions)

    @step
    def make_predictions(self):
        # Load the model from the file
        model = tf.keras.models.load_model('model_%s' % self.num_trees)
        serving_ds = tfdf.keras.pd_dataframe_to_tf_dataset(self.preprocessed_serving_df).map(self.tokenize_names)

        def prediction_to_kaggle_format(model, threshold=0.5):
            proba_survive = model.predict(serving_ds, verbose=0)[:,0]
            return pd.DataFrame({
                "PassengerId": self.serving_df["PassengerId"],
                "Survived": (proba_survive >= threshold).astype(int)
            })

        self.kaggle_predictions = prediction_to_kaggle_format(model)
        self.kaggle_predictions.to_csv('predictions.csv', index=False)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    TitanicFlow()



import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model_1000')

# Load the pre-made predictions
predictions = pd.read_csv('predictions.csv')

# Create a function to calculate accuracy
def calculate_accuracy(df, column, model):
    accuracies = []
    for value in df[column].unique():
        subset = df[df[column] == value]
        labels = subset["Survived"].values
        predictions = model.predict(subset.drop("Survived", axis=1))
        predictions = (predictions >= 0.5).astype(int)
        accuracy = (predictions == labels).mean()
        accuracies.append((value, accuracy))
    return pd.DataFrame(accuracies, columns=[column, 'Accuracy'])

# Streamlit app
st.title('Model Accuracy Analysis')

option = st.selectbox(
    'Which category do you want to analyze?',
    ('Sex', 'Pclass'))

st.write('You selected:', option)

accuracy_df = calculate_accuracy(predictions, option, model)

st.table(accuracy_df)