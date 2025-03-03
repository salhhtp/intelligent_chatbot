# Intelligent Chatbot

An Intelligent Chatbot built with Flask and Hugging Face's Transformers. This project demonstrates a conversational AI that uses a text-generation pipeline (with DialoGPT) to respond to user queries in real time. The application includes a simple web-based chat interface and a backend API for processing messages.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)
- [License](#license)

## Overview

This project showcases an intelligent chatbot that processes user input and generates responses using a pre-trained conversational model. The backend is built with Flask, and the frontend uses HTML, CSS, and JavaScript for a simple chat interface. The chatbot leverages the text-generation pipeline from Hugging Face Transformers to simulate conversation.

## Features

- **Real-time Interaction:** Users can send messages and receive responses instantly.
- **Conversational AI:** Utilizes the DialoGPT-medium model to generate context-aware responses.
- **Simple Web Interface:** A basic chat UI that displays the conversation history.
- **API-Based Architecture:** The backend exposes a POST `/ask` endpoint to process user queries.

## Tech Stack

- **Backend:** Python, Flask
- **NLP:** Hugging Face Transformers, PyTorch
- **Frontend:** HTML, CSS, JavaScript (Fetch API)
- **Deployment:** Can be deployed to Heroku, AWS, or similar platforms

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/intelligent-chatbot.git
   cd intelligent-chatbot
