# Network Security Phishing Detection MLOps

This repository contains the code for a network security phishing detection project using MLOps principles. The project includes data ingestion, model training, and deployment pipelines.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Logging](#logging)
- [Exception Handling](#exception-handling)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/NetworkSecurityPhishingDetection_MLOPS.git
    cd NetworkSecurityPhishingDetection_MLOPS
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

5. Set up environment variables:
    - Create a [`.env`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fmrsin%2FOneDrive%2FDesktop%2FNetworkSecurityPhisingDetection_MLOPS_END_TO_END%2F.env%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\mrsin\OneDrive\Desktop\NetworkSecurityPhisingDetection_MLOPS_END_TO_END\.env") file in the root directory and add the necessary environment variables, such as `MONGO_DB_URL`.

## Usage

To start the training process, run the following command:
```sh
python start_training.py


NetworkSecurityPhishingDetection_MLOPS/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   └── model_training.py
│
├── exception/
│   └── exception.py
│
├── logger/
│   └── logger.py
│
├── pipeline/
│   └── training_pipeline.py
│
├── .env
├── .gitignore
├── requirements.txt
├── start_training.py
└── README.md