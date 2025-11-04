# SharkFest_25_EU_Workshop
Code for the Talk to Your Packets Workshop

Repository for Sharfest'25 EU in Warsaw, Poland - RAG Session
John Capobianco 

## Getting Started

1. Git - Please ensure you have Git installed on your machine. You can download it from [git-scm.com](https://git-scm.com/).

2. Python - Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

3. wsl - If you are using Windows, please ensure you have WSL2 installed. You can follow the instructions [here](https://docs.microsoft.com/en-us/windows/wsl/install).

4. Ubuntu - If you are using WSL2, please ensure you have Ubuntu installed. You can follow the instructions [here](https://docs.microsoft.com/en-us/windows/wsl/install).

5. Virtual Environment - It's a good practice to create a virtual environment for your Python projects. You can do this using the following command:
   
   ```bash
   python -m venv venv
   ```

6. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

7. Install VS Code - If you haven't already, download and install Visual Studio Code from [code.visualstudio.com](https://code.visualstudio.com/).

8. Export enviroment variable 

```bash
export OPENAI_API_KEY="Key Provided By John"
```

9. Install Required Packages - Use the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

10. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/automateyournetwork/SharkFest_25_EU_Workshop
    ```

11. Navigate to the project directory:
    ```bash
    cd SharkFest_25_EU_Workshop
    ```
12. Open the project in Visual Studio Code:
    ```bash
    code .
    ```

## RAG Lab Instructions - 

Setup your venv, install requirements.txt, run the labs - change directory in to Lab01, Lab02, Lab03 etc and run the respective python files.

Some labs will be streamlit apps - run them with streamlit run app.py

## MCP Lab Instructions -

Change into the Lab01, Lab02 etc directories and run the respective python files. First run the server to start it up, stop the server, then run the client to connect to the server and execute commands.

## MacOS needs tshark
brew install --cask wireshark