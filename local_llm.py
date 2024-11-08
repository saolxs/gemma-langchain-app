from typing import List, Optional
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi import Form
import logging

# Define a custom template with placeholders for query and answer format
template = """
Question: {question}
Answer: {answer_format}
"""

# Initialize the FastAPI app
app = FastAPI(title="LangChain", version="1.0", description="The first server ever!")


# Root endpoint with a simple welcome page and form
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
                h1 {
                    color: #4CAF50;
                }
                .container {
                    width: 300px;
                    padding: 20px;
                    background-color: #2d2d2d;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
                    text-align: center;
                }
                label {
                    display: block;
                    margin-top: 10px;
                    font-size: 14px;
                }
                input, select, button {
                    width: 100%;
                    padding: 10px;
                    margin-top: 5px;
                    font-size: 14px;
                    border-radius: 4px;
                    border: none;
                }
                button {
                    background-color: #4CAF50;
                    color: white;
                    cursor: pointer;
                    margin-top: 20px;
                }
                button:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>LangChain RAG App</h1>
            <div class="container">
                <form action="/chain" method="post">
                    <label for="question">Enter Your Question:</label>
                    <input type="text" id="question" name="question" required>

                    <label for="model">Select Model:</label>
                    <select id="model" name="model" required>
                        <option value="gemma2">Gemma2</option>
                        <option value="gemma">Gemma</option>
                        <option value="llama3.1">Llama 3.1</option>
                        <option value="mistral-nemo">Mistral Nemo</option>
                        <!-- Add more model options as needed -->
                    </select>

                    <label for="answer_format">Answer Format:</label>
                    <input type="text" id="answer_format" name="answer_format"  optional>

                    <button type="submit">Submit Query</button>
                </form>
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
            <title>RAG Query Page</title>
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
            <h1>RAG Query Page</h1>
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
        add_routes(app, rag_chain, path="/")

        # Invoke the RAG chain with the formatted question and answer
        response = rag_chain.invoke(prompt_input)
        
        print(response)
        
        # Display the response in an HTML format
        return f"""
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
                        color: #4CAF50;
                        text-decoration: none;
                    }}
                </style>
            </head>
            <body>
                <h1>Model Response</h1>
                <div class="container">
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Answer Format:</strong> {answer_format}</p>
                    <p class="response"><strong>Response:</strong> {response}</p>
                    <br><a href="/">Ask Another Question</a>
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

# Define a route for the favicon to avoid 404s on favicon.ico requests
@app.get("/favicon.ico")
async def favicon():
    return {"message": "Favicon not available"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8004) #host="127.0.0.1"
