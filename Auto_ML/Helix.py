import os
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

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
            - Key metrics and their Significance
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
