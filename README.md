# VectorSec AI Security Scanner

## Overview
VectorSec AI Security Scanner is a web-based application built with Dash and Python to evaluate the security of Large Language Models (LLMs) by running predefined test cases and analyzing responses for potential vulnerabilities. The application supports multiple LLM providers (Ollama and OpenAI), provides detailed security analysis, and visualizes results through interactive dashboards.

## Features
- **Authentication**: Secure login and registration system with password hashing.
- **Test Case Management**: Loads test cases from a CSV file, categorized for easy selection.
- **LLM Integration**: Supports API calls to Ollama and OpenAI for testing LLM responses.
- **Advanced Analysis**: Uses sentiment analysis, TF-IDF vectorization, and semantic similarity to evaluate responses.
- **Interactive Dashboard**: Displays results in tables and charts, with filtering and sorting capabilities.
- **Export Options**: Generate PDF and CSV reports of test results.
- **Progress Tracking**: Real-time progress bar for test execution.
- **Responsive UI**: Built with Dash Bootstrap Components for a modern, user-friendly interface.

## Demo
Watch the [VectorSec Scanner Demo](https://youtu.be/JOCM-5WmqEU) on YouTube.

## Prerequisites
- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`):
  - dash
  - dash-bootstrap-components
  - pandas
  - plotly
  - ollama
  - requests
  - fpdf
  - nltk
  - scikit-learn
  - textblob
- A `test_cases.csv` file with columns: `Category`, `Test Case`, `Prompt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/muhammadmudassaryamin/VectorSec.git
   cd VectorSec
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure NLTK resources are downloaded:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   ```
5. Prepare the `test_cases.csv` file in the project root with the required format.

## Usage
1. Run the application:
   ```bash
   python 8poc.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:8050`.
3. Log in or register with a username and password.
4. Configure the LLM provider, model, and verifier settings.
5. Select test cases or run all tests, then view results in the dashboard.
6. Export results as PDF or CSV using the download buttons.

## File Structure
- `8poc.py`: Main application code with Dash app, LLM integration, and analysis logic.
- `test_cases.csv`: Input file containing test cases (not included; must be provided).
- `requirements.txt`: List of required Python packages.

## Configuration
- **LLM Providers**: Supports Ollama (local) and OpenAI (API key required).
- **Test Cases**: CSV file should have columns `Category`, `Test Case`, and `Prompt`.
- **Verifier Model**: Used for advanced analysis; defaults to `llama3` for Ollama.

## Security Analysis
The application performs advanced security analysis using:
- **Pattern Matching**: Detects refusal and suspicious content in responses.
- **Sentiment Analysis**: Uses NLTK's VADER to assess response tone.
- **Semantic Similarity**: Employs TF-IDF and cosine similarity to compare responses to test cases.
- **LLM Verification**: Uses a secondary LLM to validate and score responses.

## Output
- **Results Table**: Displays test case details, severity, score, response, explanation, and mitigation.
- **Charts**: Visualizes severity distribution and score histograms.
- **Reports**: Exports results as PDF or CSV for further analysis.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the Appache License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or questions, please open an issue on GitHub or contact the maintainer at [muddasaryamin@gmail.com](mailto:muddasaryamin@gmail.com).
