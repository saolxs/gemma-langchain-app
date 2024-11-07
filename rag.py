from typing import Optional, List, Tuple, Dict, Union
from fastapi import FastAPI, BackgroundTasks, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
import pandas as pd
import glob
import uvicorn
import logging
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from utils import (
    set_global_variable,
    get_text_embedding_from_text_embedding_model,
    get_document_metadata,
    get_similar_text_from_query,
    print_text_to_text_citation
)
from fastapi.staticfiles import StaticFiles
import os
from fastapi.responses import JSONResponse
import time
from langserve import add_routes

# Initialize global variables
text_metadata_df = pd.DataFrame()  # Placeholder for document embeddings
model = OllamaLLM(model="gemma2")  # Pre-load model or specify your model initialization here
processing_status = {"in_progress": False, "progress": 0, "current_file": "", "current_page": ""}

# Define a custom template with placeholders for query and answer format
template = """
Question: {question}
Answer: {answer_format}
"""

# Initialize the FastAPI app
app = FastAPI(title="LangChain", version="1.0", description="The first server ever!")

@app.get("/nogo", response_class=HTMLResponse)
async def welcome():
    return """
    <html>
        <head>
            <title>Welcome to LangChain RAG App with Gemma</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #1c1c1c;
                    color: #e0e0e0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                h1 { color: #4CAF50; }
                .container {
                    width: 300px;
                    padding: 20px;
                    background-color: #2d2d2d;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                    text-align: center;
                }
                label { display: block; margin-top: 10px; font-size: 14px; }
                input, select, button {
                    width: 100%; padding: 10px; margin-top: 5px; font-size: 14px;
                    border-radius: 4px; border: none;
                }
                button {
                    background-color: #4CAF50; color: white; cursor: pointer;
                    margin-top: 20px;
                }
                button:hover { background-color: #45a049; }
            </style>
        </head>
        <body>
            <h1>LangChain RAG App</h1>
            <div class="container">
                <form id="mainForm" action="/chain" method="post" enctype="multipart/form-data">
                    <label for="question">Enter Your Question:</label>
                    <input type="text" id="question" name="question">

                    <label for="model">Select Model:</label>
                    <select id="model" name="model">
                        <option value="gemma2">Gemma2</option>
                        <option value="gemma">Gemma</option>
                        <option value="llama3.1">Llama 3.1</option>
                        <option value="mistral-nemo">Mistral Nemo</option>
                    </select>

                    <label for="answer_format">Answer Format:</label>
                    <input type="text" id="answer_format" name="answer_format">

                    <label for="fileInput">Upload Documents:</label>
                    <input type="file" id="fileInput" name="files" multiple>

                    <button type="submit">Submit</button>
                </form>

                <div id="statusText"></div>
            </div>

            <script>
                document.getElementById("mainForm").addEventListener("submit", function(event) {
                    event.preventDefault();
                    const formData = new FormData(event.target);

                    fetch('/chain', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById("statusText").textContent = data.message;
                    })
                    .catch(error => console.error('Error:', error));
                });
            </script>
        </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def welcome_page():
    return """
    <html>
        <head>
            <title>Welcome to LangChain RAG App with Gemma</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #1c1c1c;
                    color: #e0e0e0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                h1 { color: #4CAF50; }
                .container {
                    width: 300px;
                    padding: 20px;
                    background-color: #2d2d2d;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                    text-align: center;
                }
                label { display: block; margin-top: 10px; font-size: 14px; }
                input, select, button {
                    width: 100%; padding: 10px; margin-top: 5px; font-size: 14px;
                    border-radius: 4px; border: none;
                }
                button {
                    background-color: #4CAF50; color: white; cursor: pointer;
                    margin-top: 20px;
                }
                button:hover { background-color: #218838; }
                #loadingBar {
                    width: 100%;
                    background-color: #ddd;
                    margin: 10px 0;
                }
                #loadingBarProgress {
                    width: 0;
                    height: 30px;
                    background-color: #4CAF50;
                    text-align: center;
                    color: white;
                }
                a {
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #4CAF50; /* Green background */
                    color: #ffffff; /* White text */
                    text-decoration: none; /* Remove underline */
                    border-radius: 30px; /* Circular shape for a toggle-like button */
                    text-align: center;
                    font-weight: bold;
                    transition: background-color 0.3s ease;
                }

                a:hover {
                    background-color: #218838; /* Darker green on hover */
                }

                a.active {
                    background-color: #dc3545; /* Red background for active state */
                }
            </style>
        </head>
        <body>
            <h1>LLM App</h1>
            <div class="container">
                <form action="/chain" method="post">
                    <label for="question">Enter Your Question:</label>
                    <input type="text" id="question" name="question">

                    <label for="model">Select Model:</label>
                    <select id="model" name="model">
                        <option value="gemma2">Gemma2</option>
                        <option value="gemma">Gemma</option>
                        <option value="llama3.1">Llama 3.1</option>
                        <option value="mistral-nemo">Mistral Nemo</option>
                    </select>

                    <label for="answer_format">Answer Format:</label>
                    <input type="text" id="answer_format" name="answer_format">

                    <button type="submit">Submit</button>
                </form>

                <br><a href="/rag">Use RAG </a>
                <div id="statusText"></div>
            </div>
        </body>
    </html>
    """
    

@app.get("/rag", response_class=HTMLResponse)
async def rag_welcome_page():
    return """
        <html>
        <head>
            <title>Welcome to LangChain RAG App with Gemma</title>
            <script>
                async function uploadDocuments() {
                    const files = document.getElementById("fileInput").files;
                    if (files.length === 0) {
                        alert("Please select files to upload.");
                        return;
                    }

                    const formData = new FormData();
                    for (let i = 0; i < files.length; i++) {
                        formData.append("files", files[i]);
                    }

                    // Disable upload and query button, show loading bar
                    document.getElementById("uploadButton").disabled = true;
                    document.getElementById("queryButton").disabled = true;
                    document.getElementById("loadingContainer").style.display = "block";
                    document.getElementById("progressBar").style.width = "0%";
                    document.getElementById("progressText").innerText = "Processing: 0%";

                    // Start document upload
                    await fetch("/upload_documents", {
                        method: "POST",
                        body: formData,
                    });

                    // Poll processing status
                    checkProcessingStatus();
                }

                async function checkProcessingStatus() {
                    const response = await fetch("/processing_status");
                    const status = await response.json();
                    
                    if (status.in_progress) {
                        // Update the progress bar
                        document.getElementById("progressBar").style.width = status.progress + "%";
                        document.getElementById("progressText").innerText = "Processing: " + status.progress + "%";

                        // Poll every second
                        setTimeout(checkProcessingStatus, 1000);
                    } else {
                        // Hide loading bar and enable query button
                        document.getElementById("loadingContainer").style.display = "none";
                        document.getElementById("queryButton").disabled = false;
                        document.getElementById("uploadButton").disabled = false;
                        alert("Documents processed successfully. You can now submit queries.");
                    }
                }
            </script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #1c1c1c;
                    color: #e0e0e0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                h1 { color: #4CAF50; }
                .container {
                    width: 300px;
                    padding: 20px;
                    background-color: #2d2d2d;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                    text-align: center;
                }
                label { display: block; margin-top: 10px; font-size: 14px; }
                input, select, button {
                    width: 100%; padding: 10px; margin-top: 5px; font-size: 14px;
                    border-radius: 4px; border: none;
                }
                 a {
                    display: inline-block;
                    padding: 10px 20px;
                    background-color: #4CAF50; /* Green background */
                    color: #ffffff; /* White text */
                    text-decoration: none; /* Remove underline */
                    border-radius: 30px; /* Circular shape for a toggle-like button */
                    text-align: center;
                    font-weight: bold;
                    transition: background-color 0.3s ease;
                }

                a:hover {
                    background-color: #218838; /* Darker green on hover */
                }

                a.active {
                    background-color: #dc3545; /* Red background for active state */
                }
                button {
                    background-color: #4CAF50; color: white; cursor: pointer;
                    margin-top: 20px;
                }
                button:hover { background-color: #218838; }
                #loadingBar {
                    width: 100%;
                    background-color: #ddd;
                    margin: 10px 0;
                }
                #loadingBarProgress {
                    width: 0;
                    height: 30px;
                    background-color: #4CAF50;
                    text-align: center;
                    color: white;
                }
            </style>
        </head>
        <body>
            <h1>RAG App</h1>
            <div class="container">
            <form id="uploadForm" enctype="multipart/form-data" onsubmit="event.preventDefault(); uploadDocuments();">
                <label for="fileInput">Upload PDFs:</label>
                <input type="file" id="fileInput" name="files" multiple required>
                <button id="uploadButton" type="submit">Upload</button>
            </form>

            <!-- Loading bar container -->
            <div id="loadingContainer">
                <div id="progressBar"></div>
                <div id="progressText">Processing: 0%</div>
            </div>

            <!-- Query form, initially disabled -->
            <form id="queryForm" action="/query_documents" method="post" style="margin-top: 20px;">
                <label for="question">Enter Your Question:</label>
                <input type="text" id="question" name="question" required>
                <label for="answer_format">Answer Format:</label>
                <input type="text" id="answer_format" name="answer_format" value="Provide a detailed answer based on the context" required>
                <button id="queryButton" type="submit" disabled>Submit Query</button>
            </form>
                
             <div id="loadingBar">
                    <div id="loadingBarProgress">0%</div>
                </div>
                <div id="statusText"></div>
                 <br><a href="/">Use LLM </a>
            </div>
        </body>
    </html>
    """
    
    
# Define the /chain page for question input and model selection
@app.get("/chain", response_class=HTMLResponse)
async def chain_form():
    return """
    <html>
        <head>
            <title>LLM Query Page</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #1c1c1c;
                    color: #e0e0e0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                }
                h1 {
                    color: #fff;
                }
                .container {
                    width: 300px;
                    padding: 20px;
                    background-color: #2d2d2d;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                }
                .form-label {
                    margin-top: 10px;
                    color: #a0a0a0;
                }
                input, select, button {
                    width: 100%;
                    padding: 10px;
                    margin-top: 5px;
                    margin-bottom: 15px;
                    background-color: #333;
                    color: #e0e0e0;
                    border: none;
                    border-radius: 5px;
                }
                button {
                    background-color: #4CAF50;
                    font-weight: bold;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #45a049;
                }
                .error {
                    color: #ff6b6b;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <h1>LLM Query Page</h1>
            <div class="container">
                <form action="/chain" method="post">
                    <label class="form-label" for="question">Question*</label>
                    <input type="text" id="question" name="question" required placeholder="Enter your question">
                    
                    <label class="form-label" for="model">Model Selection*</label>
                    <select id="model" name="model">
                        <option value="gemma2">Gemma2</option>
                        <option value="gemma">Gemma</option> <!-- Add other models here -->
                    </select>
                    
                    <button type="submit">Submit Query</button>
                </form>
            </div>
        </body>
    </html>
    """
    
# Main endpoint for the RAG chain
@app.post("/chain", response_class=HTMLResponse)
async def chain_endpoint(
    question: str = Form(...),
    model: str = Form(...),
    answer_format: Optional[str] = Form(...)
):
    try:
        # Set up the prompt with dynamic question and answer formatting
        prompt_template = ChatPromptTemplate.from_template(template)
        
        # Create a dictionary with the placeholders replaced
        prompt_input = {
            "question": question,
            "answer_format": answer_format
        }

        # Configure the selected model
        selected_model = OllamaLLM(model=model)
        rag_chain = prompt_template | selected_model

        # Add the chain route
        add_routes(app, rag_chain, path="/chain")

        # Invoke the RAG chain with the formatted question and answer
        response = rag_chain.invoke(prompt_input)
        
        print(response)
        
        # Display the response in an HTML format
        return f"""
        <html>
            <head>
                <title>LLM Query Response</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #1c1c1c;
                        color: #e0e0e0;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                    }}
                    h1 {{
                        color: #fff;
                    }}
                    .container {{
                        width: 300px;
                        padding: 20px;
                        background-color: #2d2d2d;
                        border-radius: 8px;
                        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                    }}
                    .response {{
                        margin-top: 15px;
                    }}
                    a {{
                        display: inline-block;
                        padding: 10px 20px;
                        background-color: #4CAF50; /* Green background */
                        color: #ffffff; /* White text */
                        text-decoration: none; /* Remove underline */
                        border-radius: 30px; /* Circular shape for a toggle-like button */
                        text-align: center;
                        font-weight: bold;
                        transition: background-color 0.3s ease;
                    }}

                    a:hover {{
                        background-color: #218838; /* Darker green on hover */
                    }}
                    a.active {{
                        background-color: #dc3545; /* Red background for active state */
                    }}
                </style>
            </head>
            <body>
                <h1>Model Response</h1>
                <div class="container">
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Answer Format:</strong> {answer_format}</p>
                    <p><strong>Model:</strong> {model}</p>
                    <p class="response"><strong>Response:</strong> {response}</p>
                    <br><a href="/">Ask Another Question</a>
                    <br></br>
                    <br><a href="/rag">Use RAG </a>
                    <br></br>
                    <br><a href="/chain/playground">Use LangServe </a>
                </div>
            </body>
        </html>
        """
    except KeyError as e:
        # Log the KeyError for missing keys
        logging.error(f"KeyError in chain_endpoint: {e}")
        return PlainTextResponse(f"KeyError: {str(e)}. Make sure 'question' and 'answer_format' keys exist.", status_code=500)
    except Exception as e:
        # Log general errors
        logging.error(f"Error in chain_endpoint: {e}")
        return PlainTextResponse(f"An error occurred: {str(e)}", status_code=500)

    # Process the question as usual if no files are uploaded
    # return {"message": f"Question submitted: {question} with model {model} and format {answer_format}"}



# Modify upload endpoint to handle processing asynchronously and set status
@app.post("/upload_documents", response_class=PlainTextResponse)
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    global processing_status
    if len(files) > 10:
        return "You can upload a maximum of 10 files at once."
    
    processing_status = {"in_progress": True, "progress": 0}
    
    # Directory where files will be temporarily saved for processing
    pdf_folder_path = "uploaded_files/"
    
    # Save files and start background processing
    for file in files:
        file_path = f"{pdf_folder_path}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
    
    
    # Set initial processing status
    processing_status = {"in_progress": True, "progress": 20, "current_file": "", "current_page": ""}
    
    # Launch processing in the background
    background_tasks.add_task(process_documents, pdf_folder_path)
    
    return JSONResponse(content={f"message": "Documents are being processed, progress: 100"})   

# Background task to process documents and update progress
def process_documents(pdf_folder_path: str):
    global processing_status, text_metadata_df
    
    # Dummy variable to simulate progress (e.g., 10 stages)
    stages = 100
    for i in range(stages):
        # Simulate each stage processing
        processing_status["progress"] = int((i + 1) / stages * 55)
    
    # Actual processing call (adjust as needed)
    text_metadata_df = get_document_metadata(
        generative_multimodal_model=model,
        pdf_folder_path=pdf_folder_path,
        image_save_dir="images",
        image_description_prompt="Provide a concise description of the image content.",
        embedding_size=768,
        add_sleep_after_document=True,
        sleep_time_after_document=5
    )
    
    for i in range(stages):
        # Simulate each stage processing
        processing_status["progress"] = int((i + 1) / stages * 75)
            
    # Update processing status on completion
    processing_status = {"in_progress": False, "progress": 75}


# Endpoint for querying the uploaded documents
@app.post("/query_documents", response_class=HTMLResponse)
async def query_documents(
    question: str = Form(...),
    answer_format: Optional[str] = Form("Provide a detailed answer based on the context")
):
    global processing_status, text_metadata_df

    stages = 100
    for i in range(stages):
            # Simulate each stage processing
            processing_status["progress"] = int((i + 1) / stages * 40)
        
    # Validate if there are any embeddings
    if text_metadata_df.empty:
        return PlainTextResponse("No documents uploaded. Please upload documents first.", status_code=400)

    # Get relevant chunks based on query
    matching_results_text = get_similar_text_from_query(
        query=question,
        text_metadata_df=text_metadata_df,
        column_name="text_embedding_chunk",  # Assuming column for embeddings is set correctly
        top_n=3,
        chunk_text=True
    )

    # Combine matched text for the context
    context = "\n".join([value["chunk_text"] for key, value in matching_results_text.items()])

    # Generate response using RAG chain
    template = f"""
    Question: {{question}}
    Context: {{context}}
    Answer: {answer_format}
    """
    prompt = ChatPromptTemplate.from_template(template)

     # Dummy variable to simulate progress (e.g., 10 stages)
    stages = 100
    for i in range(stages):
        # Simulate each stage processing
        processing_status["progress"] = int((i + 1) / stages * 70)
        
    # Set up the input data with question and context
    input_data = {
        "question": question,
        "context": context
    }

    # Invoke the chain
    rag_chain = prompt | model
    
    add_routes(app, rag_chain, path="/rag_chain")
    
    output = rag_chain.invoke(input_data)
    

    for i in range(stages):
        # Simulate each stage processing
        processing_status["progress"] = int((i + 1) / stages * 100)

    
    print(output)

    # Display the result in HTML format
    return f"""
          <html>
          <html>
            <head>
                <title>RAG Query Response</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #1c1c1c;
                        color: #e0e0e0;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                    }}
                    h1 {{
                        color: #fff;
                    }}
                    .container {{
                        width: 300px;
                        padding: 20px;
                        background-color: #2d2d2d;
                        border-radius: 8px;
                        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                    }}
                    .response {{
                        margin-top: 15px;
                    }}
                    a {{
                        display: inline-block;
                        padding: 10px 20px;
                        background-color: #4CAF50; /* Green background */
                        color: #ffffff; /* White text */
                        text-decoration: none; /* Remove underline */
                        border-radius: 30px; /* Circular shape for a toggle-like button */
                        text-align: center;
                        font-weight: bold;
                        transition: background-color 0.3s ease;
                    }}

                    a:hover {{
                        background-color: #218838; /* Darker green on hover */
                    }}
                    a.active {{
                        background-color: #dc3545; /* Red background for active state */
                    }}
                </style>
            </head>
            <body>
                <h1>Query Response</h1>
                <div class="container">
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Answer Format:</strong> {answer_format}</p>
                    <p><strong>Context:</strong> {context}</p>
                    <p><strong>Response:</strong> {output}</p>
                    <br><a href="/rag">Ask Another Question</a>
                    <br></br>
                    <br><a href="/">Use LLM </a>
                    <br></br>
                    <br><a href="/rag_chain/playground">Use LangServe </a>
                </div>
            </body>
        </html>
        """

@app.get("/processing_status")
async def get_processing_status():
    """Endpoint to check the current processing status."""
    return JSONResponse(content=processing_status)


# Define a route for the favicon to avoid 404s on favicon.ico requests
@app.get("/favicon.ico")
async def favicon():
    return {"message": "Favicon not available"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000) #host="127.0.0.1"
