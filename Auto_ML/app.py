import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import streamlit as st
import time
from io import BytesIO
from auto_prepro import Preprocessing
from auto_gen import Auto_GA_FS_Tr
from auto_reg import Regression
from auto_classfy import Classification
from sklearn.model_selection import train_test_split
import base64
from sklearn.preprocessing import StandardScaler


class Visualization:
    def __init__(self):
        pass

    def plot_correlation(self, data):
        try:
            corr = data.corr()
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Matrix")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting correlation matrix: {e}")

    def plot_histogram(self, data):
        try:
            data.hist(figsize=(20, 20))
            plt.suptitle("Histograms", x=0.5, y=0.92)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting histograms: {e}")

    def plot_boxplot(self, data):
        try:
            data.plot(kind="box", subplots=True, layout=(4, 4), figsize=(20, 20))
            plt.suptitle("Boxplot", x=0.5, y=0.92)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting boxplots: {e}")

    def plot_scatter(self, data):
        try:
            pd.plotting.scatter_matrix(data, figsize=(20, 20))
            plt.suptitle("Scatter Matrix", x=0.5, y=0.92)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting scatter matrix: {e}")

    def plot_countplot(self, data, column):
        try:
            sns.countplot(x=column, data=data)
            plt.title(f"Countplot of {column}")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting countplot: {e}")

    def plot_pairplot(self, data):
        try:
            sns.pairplot(data, diag_kind="kde")
            plt.title("Pairplot")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting pairplot: {e}")

    def plot_heatmap(self, data):
        try:
            sns.heatmap(data.corr(), annot=True)
            plt.title("Heatmap")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting heatmap: {e}")

    def plot_barplot(self, data, x, y):
        try:
            sns.barplot(x=x, y=y, data=data)
            plt.title("Barplot")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting barplot: {e}")

    def plot_confusion_matrix(self, y_true, y_pred):
        try:
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, cmap="coolwarm")
            plt.title("Confusion Matrix")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting confusion matrix: {e}")

    def plot_roc_curve(self, y_true, y_pred):
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, color="orange", label="ROC")
            plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting ROC curve: {e}")

    def plot_auc_score(self, y_true, y_pred):
        try:
            auc = roc_auc_score(y_true, y_pred)
            st.write(f"AUC: {auc}")
            return auc
        except Exception as e:
            st.error(f"Error calculating AUC score: {e}")

    def plot_lime(self, model, X_train, X_test, y_train, y_test):
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns,
                class_names=["0", "1"],
                discretize_continuous=True,
            )
            exp = explainer.explain_instance(
                X_test.values[0], model.predict, num_features=len(X_train.columns)
            )
            exp.show_in_notebook(show_table=True)
        except Exception as e:
            st.error(f"Error plotting LIME: {e}")

    def plot_feature_importance(self, model, X_train):
        try:
            importance = model.feature_importances_
            for i, v in enumerate(importance):
                st.write(f"Feature: {X_train.columns[i]}, Score: {v}")
            plt.bar([x for x in range(len(importance))], importance)
            plt.xticks(range(len(importance)), X_train.columns, rotation=90)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting feature importance: {e}")


def generate_html_report(
    dataset_name, target_variable, task_method, models, metrics, plots, preprocessing_steps, description
):
    try:
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
            <p><strong>Target Variable:</strong> {target_variable}</p>
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
    except Exception as e:
        st.error(f"Error generating HTML report: {e}")
        return ""


def save_plots(plots):
    try:
        encoded_plots = ""
        for plot in plots:
            plt.figure(plot["fig_num"])
            buf = plot["buf"]
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            encoded_plots += f'<h3>{plot["title"]}</h3><img src="data:image/png;base64,{img_base64}" />'
        return encoded_plots
    except Exception as e:
        st.error(f"Error saving plots: {e}")
        return ""


def get_plot_buffers(plots):
    try:
        buffers = []
        for plot in plots:
            buf = BytesIO()
            plt.figure(plot["fig_num"])
            plt.savefig(buf, format="png")
            buf.seek(0)
            buffers.append(
                {"fig_num": plot["fig_num"], "buf": buf, "title": plot["title"]}
            )
        return buffers
    except Exception as e:
        st.error(f"Error getting plot buffers: {e}")
        return []


def evaluate_classification(y_test, y_pred):
    try:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro"),
            "Recall": recall_score(y_test, y_pred, average="macro"),
            "F1 Score": f1_score(y_test, y_pred, average="macro"),
            "ROC AUC": roc_auc_score(y_test, y_pred),
        }
    except Exception as e:
        st.error(f"Error evaluating classification: {e}")
        return {}


def evaluate_regression(y_test, y_pred):
    try:
        return {
            "Mean Squared Error": mean_squared_error(y_test, y_pred),
            "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred),
        }
    except Exception as e:
        st.error(f"Error evaluating regression: {e}")
        return {}


# Streamlit App
st.title("Helix AI - Supervised Learning Module")

# Upload Dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
description = st.text_area("Dataset Description", "Enter a brief description of the dataset.")

    
# Initialize a string to store steps
steps=f"Dataset Description:\n{description}\n\n"

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        steps += f"First few rows of the data:\n{data.head().to_string()}\n\n"

        
        # User Input for Target Column
        target_column = st.selectbox("Select the target column", data.columns)

        # User Input for Task Type
        task_type = st.selectbox("Select Task Type", ["Regression", "Classification"])

        # User Input for Model Selection
        model_types = []
        if task_type == "Regression":
            model_types = st.multiselect(
                "Select Regression Models",
                [
                    "linear",
                    "ridge",
                    "lasso",
                    "elasticnet",
                    "bayesianridge",
                    "ransac",
                    "svr",
                    "decisiontree",
                    "randomforest",
                    "adaboost",
                    "gradientboosting",
                    "mlp",
                    "knn",
                    "gaussianprocess",
                ],
            )
            reg = Regression()
        elif task_type == "Classification":
            model_types = st.multiselect(
                "Select Classification Models",
                [#
                    "logistic",
                    "decisiontree",
                    "randomforest",
                    "svm",
                    "knn",
                    "mlp",
                    "adaboost",
                    "gradientboosting",
                    "gaussianprocess",
                ],
            )
            clf = Classification()

        # Preprocessing
        prepro = Preprocessing()

        # Visualizations
        vis = Visualization()

        # User input for which plots to generate
        plot_options = st.multiselect(
            "Select Plots to Generate",
            [
                "Correlation Matrix",
                "Histograms",
                "Boxplots",
                "Scatter Matrix",
                "Countplot",
                "Heatmap",
                "Pairplot",
            ],
            default=["Correlation Matrix", "Histograms", "Boxplots"],
        )

        if st.button("Preprocess and Visualize"):
            # Start timer
            start_time = time.time()

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Drop target column from feature set if not clustering
            if task_type != "Clustering":
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                X = data
                y = None

            try:
                # Scaling the data
                progress_bar.progress(10)
                status_text.text("Scaling the data...")
                scaled_data = prepro.scale_data(X, method="standard")

                # PCA
                progress_bar.progress(20)
                status_text.text("Applying PCA...")
                pca_data = prepro.apply_pca(scaled_data, n_components=2)

                # Encoding categorical features
                progress_bar.progress(30)
                status_text.text("Encoding categorical features...")
                if any(data.dtypes == "object"):
                    categorical_columns = data.select_dtypes(include=["object"]).columns
                    encoded_data = prepro.encode_categorical(
                        scaled_data, columns=categorical_columns
                    )
                else:
                    encoded_data = scaled_data

                # Imputing missing values
                progress_bar.progress(40)
                status_text.text("Imputing missing values...")
                imputed_data = prepro.impute_missing_values(encoded_data)

                # Train Test Split if not clustering
                if task_type != "Clustering":
                    progress_bar.progress(50)
                    status_text.text("Splitting the data into train and test sets...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        imputed_data, y, test_size=0.2, random_state=42
                    )
                else:
                    X_train, X_test, y_train, y_test = imputed_data, None, None, None
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")

            # GA-based Feature Selection and Training
            progress_bar.progress(60)
            status_text.text("Performing GA-based feature selection and training...")

            metrics = {}
            try:
                for model_type in model_types:
                    if task_type == "Regression":
                        model = reg.auto_regressor(model_type)
                    elif task_type == "Classification":
                        model = clf.auto_classify(model_type)

                    ga_fs_tr = Auto_GA_FS_Tr(
                        model=model,
                        param_grid={},
                        X_train=X_train.to_numpy(),
                        y_train=(y_train.to_numpy() if y_train is not None else None),
                        X_test=(X_test.to_numpy() if X_test is not None else None),
                        y_test=(y_test.to_numpy() if y_test is not None else None),
                        population_size=10,
                        generations=10,
                        mutation_rate=0.1,
                        type_method=task_type.lower(),
                    )
                    best_individual, best_fitness = ga_fs_tr.fit()
                    st.write(
                        f"Best Individual Features for {model_type}: ", best_individual
                    )
                    st.write(f"Best Fitness for {model_type}: ", best_fitness)

                    # Model Evaluation
                    if task_type == "Classification":

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        metrics[model_type] = evaluate_classification(y_test, y_pred)
                    elif task_type == "Regression":

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        metrics[model_type] = evaluate_regression(y_test, y_pred)

            except Exception as e:
                st.error(f"Error during model training and evaluation: {e}")

            # Visualizations
            progress_bar.progress(70)
            status_text.text("Generating visualizations...")

            plot_funcs = {
                "Correlation Matrix": vis.plot_correlation,
                "Histograms": vis.plot_histogram,
                "Boxplots": vis.plot_boxplot,
                "Scatter Matrix": vis.plot_scatter,
                "Countplot": vis.plot_countplot,
                "Heatmap": vis.plot_heatmap,
                "Pairplot": vis.plot_pairplot,
            }

            plot_progress = 70
            plot_increment = 30 // len(plot_options)
            plots = []

            for plot in plot_options:
                try:
                    if plot == "Countplot":
                        if (
                            data[target_column].dtype == "object"
                            or len(data[target_column].unique()) <= 20
                        ):
                            st.write(f"{plot}")
                            plot_funcs[plot](data, column=target_column)
                            buf = BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            plots.append(
                                {
                                    "fig_num": plot_progress // plot_increment,
                                    "buf": buf,
                                    "title": plot,
                                }
                            )
                        else:
                            st.write(
                                f"{plot} not available for {target_column} due to high cardinality or non-categorical data."
                            )
                    else:
                        st.write(f"{plot}")
                        plot_funcs[plot](data)
                        buf = BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        plots.append(
                            {
                                "fig_num": plot_progress // plot_increment,
                                "buf": buf,
                                "title": plot,
                            }
                        )

                    plot_progress += plot_increment
                    progress_bar.progress(plot_progress)
                except Exception as e:
                    st.error(f"Error generating {plot}: {e}")

            # End timer
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.success(
                f"Preprocessing and visualization complete! It took {elapsed_time:.2f} seconds."
            )

            # Generate and Save HTML Report
            try:
                encoded_plots = save_plots(plots)
                html_report = generate_html_report(
                    dataset_name=uploaded_file.name,
                    target_variable=target_column,
                    task_method=task_type,
                    models=model_types,
                    metrics=metrics,
                    plots=encoded_plots,
                    description=description,
                    preprocessing_steps="Standard scaling, PCA, encoding categorical features, imputing missing values",
                )
                html_bytes = html_report.encode()

                st.download_button(
                    label="Download Report as HTML",
                    data=html_bytes,
                    file_name="auto_ml_report.html",
                    mime="text/html",
                )
                
                # appending the data.head(), data description, model used, and the metrics and preprocessing steps to the steps string
                steps += f"Data Description:\n{data.describe().to_string()}\n\n"
                steps += f"Model Used:\n{model_types}\n\n"
                steps += f"Metrics:\n{metrics}\n\n"
                steps += f"Preprocessing Steps:\nStandard scaling, PCA, encoding categorical features, imputing missing values\n\n"
                
                # showing the steps string
                st.write(steps)
                
                
            except Exception as e:
                st.error(f"Error generating HTML report: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")