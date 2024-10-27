import gradio as gr
import requests
import os
from time import perf_counter
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from pydantic import Field, field_validator
from langchain import PromptTemplate  # Langchain Prompt Template
from langchain.chains import LLMChain  # Langchain Chains
from langchain.document_loaders import PyPDFLoader
from termcolor import colored
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from langchain_experimental.text_splitter import SemanticChunker
from langchain.llms.base import LLM
from requests.auth import HTTPBasicAuth
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain, ReduceDocumentsChain

# Load the environment variables
load_dotenv()

ibm_cloud_url = "https://us-south.ml.cloud.ibm.com"
project_id = os.environ['PROJECT_ID']
api_key = os.environ['API_KEY']

if api_key is None or ibm_cloud_url is None or project_id is None:
    raise Exception("One or more environment variables are missing!")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key
    }

def model_initialize(model_name):
    # Initialize Watsonx model
    params = {
        "decoding_method": "greedy",
        "repetition_penalty": 1.1,
        "min_new_tokens": 1,
        "max_new_tokens": 1024
    }

    llm_model = Model(
        model_id=model_name,
        params=params,
        credentials=creds,
        project_id=project_id
    )
    return llm_model

def get_embedding(embedding_model):
    embed_params = {
        "truncate_input_tokens": 3,
        "return_options": {
            'input_text': True
        }
    }

    embedding = Embeddings(
        model_id=embedding_model,
        params=embed_params,
        credentials=Credentials(
            api_key=api_key,
            url="https://us-south.ml.cloud.ibm.com"
        ),
        project_id=project_id
    )
    return embedding

def make_semantic_chunking(embedding, documents, threshold_type):
    semantic_chunker = SemanticChunker(embedding, breakpoint_threshold_type=threshold_type)
    docs = semantic_chunker.create_documents([d.page_content for d in documents])
    return docs

# Define Custom Watson LLM class
class CustomWatsonLLM(LLM):
    model: object = Field(...)  

    def __init__(self, model):
        super().__init__(model=model)

    def _call(self, prompt, stop=None):
        response = self.model.generate(prompt)
        return response['results'][0]['generated_text']
    
    @property
    def _llm_type(self):
        return "custom_watson_llm"

# Function to get IBM Cloud access token
def get_access_token():
    IAM_URL = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    auth = HTTPBasicAuth("bx", "bx")
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    response = requests.post(IAM_URL, data=data, headers=headers, auth=auth)
    json_data = response.json()
    access_token = json_data['access_token']
    return access_token

# Define a function to load and summarize the PDF
def summarize_pdf(pdf_file, llm_model_name, embedding_model_name):    
    try:
        if pdf_file is None:
            raise ValueError("No PDF file uploaded.")

        # Check if the uploaded file has a 'name' attribute
        if not hasattr(pdf_file, 'name'):
            raise ValueError("Invalid file uploaded. Please upload a valid PDF file.")

        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        # Obtain access token
        get_access_token()

        # Initialize embeddings and chunker
        embedding = get_embedding(embedding_model_name)
        semantic_chunks = make_semantic_chunking(embedding, documents, "percentile")
        
        # Initialize the Watson LLM model
        llm_model = model_initialize(llm_model_name)
        custom_llm = CustomWatsonLLM(llm_model)
        
        # Map Phase
        map_template = """The following is a set of parts within one long paper
        {docs}
        Based on the list of parts of the paper, please identify the main point in less than 3 sentences.
        Start the summarization with "The main point of this section is"
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=custom_llm, prompt=map_prompt)
        
        # Reduce Phase
        reduce_template = """The following is a set of summaries:
        {doc_summaries}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Start the finalized summarization with "The final summarization of this paper is"
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=custom_llm, prompt=reduce_prompt)
        
        # Stuff documents using reduce chain
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )

        # Reduce documents chain
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000
        )

        # MapReduce chain setup
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=True,
            verbose=False
        )

        # Prepare output variable
        output_html = ""

        # Run the map phase
        output_html += "<p style='color:white;'>Running map phase...</p>"
        yield output_html

        map_results = []
        i = 1
        for chunk in semantic_chunks:
            # Assuming each chunk is a Document object with the paper content
            result = map_chain({"docs": chunk.page_content})
            map_results.append(result)
            intermediate_result = f"Intermediate map result {i}: {result['text']}"
            output_html += f"<p style='color:green;'>{intermediate_result}</p>"
            yield output_html  # Update the UI with the new intermediate result
            i += 1
        
        output_html += "<p style='color:white;'>Running reduce phase...</p>"
        yield output_html
        
        t1_start = perf_counter()
        results = map_reduce_chain(semantic_chunks)
        output = results["output_text"]
        t1_stop = perf_counter()
        elapsed_time = f"Elapsed time for reduce phase: {round((t1_stop - t1_start), 2)} seconds.\n"
        output_html += f"<p style='color:white;'>{elapsed_time}</p>"

        # Add final output to HTML
        output_html += f"<p style='color:cyan;'>{output}</p>"
        yield output_html
    except Exception as e:
        yield f"<p style='color:red;'>An error occurred during summarization: {str(e)}</p>"



# Gradio Interface with Sidebar and Scrollable Results
with gr.Blocks(css="""
    #results-box {
        height: 400px; /* Set the fixed height */
        overflow-y: scroll !important; /* Forces scrolling when content overflows */
        border: 1px solid #ccc;
        padding: 10px;
        width: 100%; /* Ensures it uses the available space */
        white-space: pre-wrap; /* Wraps text */
        word-wrap: break-word; /* Ensures long words break */
    }
""") as demo:
    with gr.Row():
        with gr.Column(visible=True, min_width=200, scale=0) as sidebar:
            # Sidebar components
            pdf_file = gr.File(label="Upload a PDF file")
            llm_model_selection = gr.Radio(["meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-8b-instruct", "meta-llama/llama-3-1-8b-instruct"], label="LLM Model", elem_classes="column-form")
            embedding_model_selection = gr.Radio(["intfloat/multilingual-e5-large", "ibm/slate-125m-english-rtrvr-v2", "sentence-transformers/all-minilm-l12-v2"], label="Embedding Model", elem_classes="column-form")
            summarize_button = gr.Button("Summarize")
        with gr.Column() as main:
            # Title and description
            gr.Markdown("# PDF Summarizer with Sidebar")
            gr.Markdown("## Select options from the sidebar and upload a PDF to summarize.")
            gr.Markdown("The summary will appear here once you click the 'Summarize' button.")
            
            # Scrollable HTML component
            result = gr.HTML(elem_id="results-box")
            
            # JavaScript to scroll to the bottom
            gr.HTML("""
                <script>
                // Function to scroll to the bottom of the results box
                function scrollToBottom() {
                    var element = document.getElementById('results-box');
                    element.scrollTop = element.scrollHeight;
                }

                // Listen to changes on the result box and scroll to the bottom
                const observer = new MutationObserver(scrollToBottom);
                observer.observe(document.getElementById('results-box'), { childList: true });
                </script>
            """)
            
            # Connect the summarize_pdf function to the button click
            summarize_button.click(
                fn=summarize_pdf, 
                inputs=[pdf_file, llm_model_selection, embedding_model_selection], 
                outputs=result
            )

if __name__ == "__main__":
    demo.launch()
    # If you want to make shareable link, uncomment the line below
    # demo.launch(share=True)