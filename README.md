# 🚀 Mekari Chatbot

This repository contains a Python application built using **Streamlit**.
Follow the instructions below to set up and run the project on your
local machine quickly.

## 📋 Prerequisites

Ensure you have the following installed: - **Python** (version 3.9 or
higher recommended) - **Git** (for cloning the repository)

## 🛠️ Installation Guide

### 1. Clone the repository

``` bash
git clone https://github.com/wahyudwi99/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment

Using a virtual environment is highly recommended.

#### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

## ▶️ How to Run the Application

The main entry point for the Streamlit app is `app.py`.

Run the application using:

``` bash
streamlit run app.py
```

The app will automatically launch in your default browser at:

**http://localhost:8501**

## 📂 Repository Structure

  -----------------------------------------------------------------------
  File/Folder                                    Description
  ---------------------------------------------- ------------------------
  **app.py**                                     Main Streamlit
                                                 application file.

  **module.py**                                  Contains reusable logic
                                                 and functions.

  **template_prompt.py**                         Stores predefined prompt
                                                 templates (e.g., for
                                                 APIs or LLMs).

  **queries/**                                   Directory for storing
                                                 query or data files.

  **requirements.txt**                           Lists all Python
                                                 dependencies.

  **.gitignore**                                 Specifies files/folders
                                                 ignored by Git (such as
                                                 venv).
  -----------------------------------------------------------------------

## 💡 Contributing

Contributions are welcome! Please open an issue or submit a Pull
Request.
