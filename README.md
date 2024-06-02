# Text-Classification---Bachelor-Thesis

## Emotion Classification App

This application classifies text into six emotional categories using a pre-trained BERT model. The available categories are Sadness, Joy, Love, Anger, Fear, and Surprise. The application also summarizes long texts before classification.

### Features

- `Text Classification`: Classify text into one of six emotional categories.
- `Text Summarization`: Automatically summarize texts longer than 512 tokens.
- `User Input`: Enter text directly or upload a text file.
- `Interactive UI`: View the classification results in an easy-to-read format with graphical representation.

### Getting Started

These instructions will help you set up and run the application on your local machine.

#### Prerequisites

- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/cli/pip_download/)
- [Git](https://git-scm.com/downloads)

#### Installation

1. Clone the Repository

```bash
git clone https://github.com/SlavkoMuzdeka/Text-Classification---Bachelor-Thesis.git
cd Text-Classification---Bachelor-Thesis
```

2. Create a Virtual Environment

It's recommended to create a virtual environment to manage dependencies. You can use venv for this.

```bash
python -m venv env
source env/bin/activate    # On Windows: env\Scripts\activate
```

3. Install Dependencies

Install the required Python packages using pip.

```bash
pip install -r requirements.txt
```

#### Running the Application

1. Run the Streamlit App

Start the Streamlit application using the following command:

```bash
streamlit run streamlit_app.py
```

2. Access the Application

Open your web browser and go to http://localhost:8501 to access the application.

### Usage

#### How to Use

- `Enter your text`: Type or paste your text into the provided text area.
- `Upload a text file`: Upload a .txt file containing the text you want to classify.
- `Classify`: Click the "Classify" button to see the classification results.

#### Sidebar Information

- `How to Use`: Instructions on how to provide text input and classify it.
- `About`: Information about the app and the classification categories.

### Contributing

If you would like to contribute to the project, please fork the repository and submit a pull request.
