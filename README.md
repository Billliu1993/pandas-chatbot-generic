# Project Setup Instructions

This project uses sample data from [Kaggle: Transaction Data](https://www.kaggle.com/datasets/vipin20/transaction-data).

## Setup Steps

To get started with the project, follow these steps under the root directory:

1. **Create a `.env` file** with the following content:
   ```dotenv
   OPENAI_API_KEY=""
   ```

2. **Create a `.streamlit` folder**, and then create a `config.toml` file inside it with the following content:
   ```toml
   [browser]
   gatherUsageStats = false
   ```

By following these steps, you'll set up the necessary environment variables and configurations to run the project.