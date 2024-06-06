import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np
import base64
import io
from Helix import multi_llm_user

class BertSentimentClassifier:
    def __init__(self, train_dataset, val_dataset, batch_size=8, max_len=128, num_epochs=1, learning_rate=2e-5):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        num_classes = len(train_dataset.df['sentiment'].unique())
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            total_loss = 0
            st.write(f'Epoch {epoch + 1}/{self.num_epochs}')
            progress_bar = st.progress(0)
            for i, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', unit='batches')):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(train_loader))
                
            avg_train_loss = total_loss / len(train_loader)
            st.write(f'Average training loss: {avg_train_loss}')
            return avg_train_loss

    def evaluate(self):
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        total_val_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', unit='batches'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.logits, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(outputs.logits.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = total_correct / total_samples

        st.write(f'Average validation loss: {avg_val_loss}')
        st.write(f'Accuracy: {accuracy}')

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        st.write("Confusion Matrix:")
        st.write(cm)

        # Precision, Recall, F1-Score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        st.write(f'Precision: {precision}')
        st.write(f'Recall: {recall}')
        st.write(f'F1-Score: {f1}')

        # Convert all_probs to a NumPy array
        all_probs = np.array(all_probs)

        # ROC Curve and AUC for each class
        all_labels_binarized = label_binarize(all_labels, classes=[0, 1, 2])
        n_classes = all_labels_binarized.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) to Multi-Class')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        st.pyplot(plt)

        # Confusion Matrix Plot
        plt.figure()
        plt.matshow(cm, cmap='coolwarm')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix.png')
        st.pyplot(plt)

        # Generate HTML report
        report = f"""
        <html>
        <head>
            <title>Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                }}
                h2 {{
                    color: #2e6c80;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .center {{
                    text-align: center;
                }}
                img {{
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                }}
            </style>
        </head>
        <body>
            <h1>Helix AI - NLP Module Report</h1>
            <h2>Dataset Description</h2>
            <h2>Evaluation Metrics</h2>
            <p><strong>Average Validation Loss:</strong> {avg_val_loss}</p>
            <p><strong>Accuracy:</strong> {accuracy}</p>
            <p><strong>Precision:</strong> {precision}</p>
            <p><strong>Recall:</strong> {recall}</p>
            <p><strong>F1-Score:</strong> {f1}</p>
            <h2>Confusion Matrix</h2>
            {pd.DataFrame(cm).to_html()}
            <h2>ROC Curve</h2>
            <img src="data:image/png;base64,{self.fig_to_base64('roc_curve.png')}" alt="ROC Curve">
            <h2>Confusion Matrix Plot</h2>
            <img src="data:image/png;base64,{self.fig_to_base64('confusion_matrix.png')}" alt="Confusion Matrix">
        </body>
        </html>
        """
        st.markdown(report, unsafe_allow_html=True)
        
        return avg_val_loss, accuracy, precision, recall, f1, report

    def fig_to_base64(self, fig_path):
        with open(fig_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, 'sentence']
        label = self.df.loc[idx, 'sentiment']
        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
def main():
    st.title("Helix AI - NLP Module Powered by BERT")
    
    # Text box for dataset description
    dataset_description = st.text_area("Dataset Description", "Enter the description of your dataset here...")
    
    # Initialize a string to store steps
    steps = f"Dataset Description:\n{dataset_description}\n\n"

    # File upload for dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, names=["sentence", 'sentiment'], encoding='ISO-8859-1')

        st.write(df.head())
        steps += "Data Loaded\n"
        steps += f"First few rows of the data:\n{df.head().to_string()}\n\n"

        # Preprocess the dataset
        st.header("Preprocessing")
        sentiment_map = {
            'positive': 2,
            'negative': 0,
            'neutral': 1
        }
        df['sentiment'] = df['sentiment'].map(sentiment_map)
        df = df.dropna(subset=['sentiment'])
        df['sentiment'] = df['sentiment'].astype(int)
        steps += "Data Preprocessed\n"

        # Train-test split
        train_df, val_df = train_test_split(df, test_size=0.2)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        steps += "Data Split into Training and Validation Sets\n"

        # Tokenization
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = CustomDataset(train_df, tokenizer, max_len=128)
        val_dataset = CustomDataset(val_df, tokenizer, max_len=128)

        # Train the model
        st.header("Training")
        ans = BertSentimentClassifier(train_dataset=train_dataset, val_dataset=val_dataset)
        avg_train_loss = ans.train()
        steps += f"Model Trained\nAverage Training Loss: {avg_train_loss}\n"

        # Evaluation
        st.header("Evaluation")
        avg_val_loss, accuracy, precision, recall, f1, report = ans.evaluate()
        steps += f"Model Evaluated\n"
        steps += f"Average Validation Loss: {avg_val_loss}\n"
        steps += f"Accuracy: {accuracy}\n"
        steps += f"Precision: {precision}\n"
        steps += f"Recall: {recall}\n"
        steps += f"F1-Score: {f1}\n"

        # Display all steps
        st.header("Summary of Steps")
        if st.button("Show Summary"):
            st.text(steps)

        # Create a download button for the HTML report
        st.download_button(
            label="Download Report",
            data=report,
            file_name="evaluation_report.html",
            mime="text/html"
        )
        # Helix AI LLM Analysis
        st.header("Helix AI - LLM Analysis")
        llm_output=multi_llm_user(data=steps)
        st.write(llm_output)
        
if __name__ == '__main__':
    main()
