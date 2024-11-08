from typing import List, Optional
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
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

   
