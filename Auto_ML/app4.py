import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc

class Visualization:
    def __init__(self):
        pass

    def plot_correlation(self, data):
        corr = data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf

    def plot_histogram(self, data):
        data.hist(figsize=(20, 20))
        plt.suptitle("Histograms", x=0.5, y=0.92)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf

    def plot_boxplot(self, data):
        num_columns = data.shape[1]
        num_rows = (
            num_columns + 3
        ) // 4  # Ensure enough rows to accommodate all columns
        data.plot(
            kind="box", subplots=True, layout=(num_rows, 4), figsize=(20, 5 * num_rows)
        )
        plt.suptitle("Boxplot", x=0.5, y=0.92)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf

    def plot_scatter(self, data):
        pd.plotting.scatter_matrix(data, figsize=(20, 20))
        plt.suptitle("Scatter Matrix", x=0.5, y=0.92)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf
    
    def plot_roc_curve(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf
    
    def plot_precision_recall_curve(self, y_true, y_pred):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        average_precision = average_precision_score(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return buf


class NeuralNetwork:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
        
    def predict(self, X):
        return self.model.predict(X)

def generate_html_report(dataset_name, task_method, models, metrics, plots, description, preprocessing_steps):
    first_model = list(metrics.keys())[0] if metrics else None
    metric_names = metrics[first_model].keys() if first_model else []

    html_content = f"""
    <html>
    <head>
        <title>Auto ML Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 50px;
                line-height: 1.6;
                color: #333;
            }}
            h1, h2, h3, p {{
                margin-bottom: 20px;
            }}
            h1 {{
                color: #4CAF50;
            }}
            h2 {{
                color: #008CBA;
            }}
            h3 {{
                color: #f44336;
            }}
            .plot {{
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Auto ML Report</h1>
        <h2>Dataset Information</h2>
        <p><strong>Dataset Name:</strong> {dataset_name}</p>
        <p><strong>Description:</strong> {description}</p>
        <p><strong>Task Method:</strong> {task_method}</p>
        <h2>Preprocessing Steps</h2>
        <p>{preprocessing_steps}</p>
        <h2>Model Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    {"".join([f"<th>{metric}</th>" for metric in metric_names])}
                </tr>
            </thead>
            <tbody>
                {''.join([f"<tr><td>{model}</td>" + "".join([f"<td>{value:.4f}</td>" for value in metrics[model].values()]) + "</tr>" for model in models]) if metrics else "<tr><td colspan='100%'>No models evaluated.</td></tr>"}
            </tbody>
        </table>
        <h2>Visualizations</h2>
        <div class="plot">{plots}</div>
    </body>
    </html>
    """
    return html_content

def save_plots(plots):
    encoded_plots = ""
    for plot in plots:
        img_base64 = base64.b64encode(plot["buf"].getvalue()).decode()
        encoded_plots += (
            f'<h3>{plot["title"]}</h3><img src="data:image/png;base64,{img_base64}" />'
        )
    return encoded_plots

def main():
    st.title("Helix AI - Neural Network  Module")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    description = st.text_area("Dataset Description", "Enter a brief description of the dataset.")
    # Initialize a string to store steps
    steps=f"Dataset Description:\n{description}\n\n"
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        st.write("Data Preview:", data.head())

        plot_types = ["Correlation Matrix", "Histogram", "Boxplot", "Scatter Matrix", "Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
        selected_plots = st.multiselect("Select the plots you need", plot_types)

        vis = Visualization()
        plots = []

        target_column = st.selectbox("Select the target column", data.columns)

        if st.button("Run Neural Network"):
            if target_column not in data.columns:
                st.error(f"Target column {target_column} is not in the data")
                return

            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Encode target if necessary
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            y = OneHotEncoder(sparse=False).fit_transform(y.values.reshape(-1, 1))

            # Standardize features
            X = StandardScaler().fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            nn = NeuralNetwork(input_dim=X.shape[1], output_dim=y.shape[1])
            nn.train(X_train, y_train, epochs=10, batch_size=32)

            loss, accuracy = nn.evaluate(X_test, y_test)
            st.write(f"Test Loss: {loss:.4f}")
            st.write(f"Test Accuracy: {accuracy:.4f}")

            preprocessing_steps = "Standard scaling the data and one-hot encoding the target"

            metrics = {"Neural Network": {"Loss": loss, "Accuracy": accuracy}}

            for plot_type in selected_plots:
                if plot_type == "Correlation Matrix":
                    buf = vis.plot_correlation(data)
                    plots.append({"title": "Correlation Matrix", "buf": buf})
                elif plot_type == "Histogram":
                    buf = vis.plot_histogram(data)
                    plots.append({"title": "Histograms", "buf": buf})
                elif plot_type == "Boxplot":
                    buf = vis.plot_boxplot(data)
                    plots.append({"title": "Boxplots", "buf": buf})
                elif plot_type == "Scatter Matrix":
                    buf = vis.plot_scatter(data)
                    plots.append({"title": "Scatter Matrix", "buf": buf})
                elif plot_type == "Confusion Matrix":
                    y_pred = nn.predict(X_test).argmax(axis=1)
                    buf = vis.plot_confusion_matrix(y_test.argmax(axis=1), y_pred)
                    plots.append({"title": "Confusion Matrix", "buf": buf})
                elif plot_type == "ROC Curve":
                    y_pred = nn.predict(X_test)
                    buf = vis.plot_roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
                    plots.append({"title": "ROC Curve", "buf": buf})
                elif plot_type == "Precision-Recall Curve":
                    y_pred = nn.predict(X_test)
                    buf = vis.plot_precision_recall_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1))
                    plots.append({"title": "Precision-Recall Curve", "buf": buf})
                                        
                    

            encoded_plots = save_plots(plots)
            html_report = generate_html_report(
                dataset_name=uploaded_file.name,
                task_method="Neural Network",
                models=["Neural Network"],
                metrics=metrics,
                plots=encoded_plots,
                description=description,
                preprocessing_steps=preprocessing_steps
            )

            st.markdown(html_report, unsafe_allow_html=True)

            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="report.html">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # appending the data.head(), data description, model used, and the metrics and preprocessing steps to the steps string

            steps += f"Data Head:\n{data.head()}\n\n"
            steps += f"Model: Neural Network\n"
            steps += f"Metrics: {metrics}\n"
            steps += f"Preprocessing Steps: {preprocessing_steps}\n"
            steps +=f"Model Summary: {nn.model.summary()}\n"
            steps +=f"Model Training Loss: {loss}\n"
            steps +=f"Model Training Accuracy: {accuracy}\n"
            steps +=f"Model Evaluation Loss: {loss}\n"
            steps +=f"Model Evaluation Accuracy: {accuracy}\n"
            steps +=f"Model Evaluation Metrics: {metrics}\n"
            
            st.write(steps)

if __name__ == "__main__":
    main()
