import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import streamlit as st
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from auto_cluster import Clustering

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

def evaluate_clustering(X, labels):
    if len(np.unique(labels)) == 1:
        return {"Silhouette Score": None}
    return {"Silhouette Score": silhouette_score(X, labels)}

def main():
    st.title("Helix AI - Unsupervised Learning Module")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    description = st.text_area("Dataset Description", "Enter a brief description of the dataset.")

    steps=f"Dataset Description:\n{description}\n\n"
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        st.write("Data Preview:", data.head())

        plot_types = ["Correlation Matrix", "Histogram", "Boxplot", "Scatter Matrix"]
        selected_plots = st.multiselect("Select the plots you need", plot_types)

        clustering_algorithms = ["kmeans", "agglomerative", "dbscan", "gmm"]
        selected_algorithms = st.multiselect(
            "Select the clustering algorithms", clustering_algorithms
        )
        num_clusters = st.slider(
            "Select the number of clusters (for applicable algorithms)", 2, 10, 3
        )

        if st.button("Run Clustering"):
            vis = Visualization()
            plots = []

            progress_bar = st.progress(0)
            progress_text = st.empty()
            total_steps = len(selected_plots) + len(selected_algorithms)
            current_step = 0

            # Generate selected plots
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
                current_step += 1
                progress_text.text(f"Generating plots: {current_step}/{total_steps}")
                progress_bar.progress(current_step / total_steps)

            # Preprocessing steps
            preprocessing_steps = "Standard scaling the data"

            # Run selected clustering algorithms
            metrics = {}
            cluster = Clustering()
            for algorithm in selected_algorithms:
                labels, model = cluster.auto_cluster(
                    data, method=algorithm, n_clusters=num_clusters
                )
                metrics[algorithm] = evaluate_clustering(data, labels)
                current_step += 1
                progress_text.text(
                    f"Running clustering algorithms: {current_step}/{total_steps}"
                )
                progress_bar.progress(current_step / total_steps)

            encoded_plots = save_plots(plots)
            html_report = generate_html_report(
                dataset_name=uploaded_file.name,
                task_method="Clustering",
                models=selected_algorithms,
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

            steps+=f"Data Preview: {data.head()} \n"
            steps+=f"Description: {description} \n"
            steps+=f"Model Used: {selected_algorithms} \n"
            steps+=f"Metrics: {metrics} \n"
            steps+=f"Preprocessing Steps: {preprocessing_steps} \n"
            
        # showing to user
        st.write(steps)
            
            
if __name__ == "__main__":
    main()
