import os
from flask import Flask, request, render_template_string
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

app = Flask(__name__)

def multi_llm_user(data):
    try:
        # Set Hugging Face API KEY
        os.environ["API_KEY"] = "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"

        # Model ID for Hugging Face
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        
        # HuggingFaceHub initialization
        falcon_llm = HuggingFaceHub(
            huggingfacehub_api_token=os.environ["API_KEY"],
            repo_id=model_id,
            model_kwargs={"temperature": 0.2, "max_new_tokens": 2000},
        )
        
        # Template for prompt
        template = """
            You are an AI assistant that generates detailed analysis from a given passage and provides a concise and relevant response. Analyze the data below and extract key insights, highlight significant points, and provide an overall summary.
            
            Data to analyze:
            {data}
            
            Your analysis should include:
            - Key metrics and their significance
            - Summary of the overall sentiment
            - Any notable trends or patterns
            - Observations about the training and validation process
            
            Your response:
        """
        
        # Prompt template initialization
        prompt = PromptTemplate(template=template, input_variables=["data"])
        
        # Combining prompt and llm using LLMChain
        falcon_chain = LLMChain(llm=falcon_llm, prompt=prompt, verbose=True)
        
        # Running the data through the chain and returning the output
        output = falcon_chain.run(data)
        return output
    
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        data = request.form['data']
        if data:
            result = multi_llm_user(data)
        else:
            result = "Please enter some data to analyze."
    
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Data Analysis with LangChain and HuggingFace</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 800px;
                    margin: auto;
                    background: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                }
                h1 {
                    text-align: center;
                    color: #333;
                }
                form {
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }
                textarea {
                    padding: 10px;
                    font-size: 16px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                }
                input[type="submit"] {
                    background-color: #007bff;
                    color: white;
                    padding: 10px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                h2 {
                    margin-top: 30px;
                    color: #333;
                }
                pre {
                    background: #f4f4f9;
                    padding: 10px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Helix-AI</h1>
                <form method="post" enctype="multipart/form-data">
                    <textarea name="data" rows="10" cols="80" placeholder="Enter the data to be analyzed"></textarea>
                    <input type="submit" value="Analyze">
                </form>
                <h2>Analysis Result</h2>
                <pre>{{ result }}</pre>
            </div>
        </body>
        </html>
    ''', result=result)

if __name__ == "__main__":
    app.run(debug=True)
