# Family Medicine Clinical Practice Guidelines Chatbot
## Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot designed to assist licensed healthcare providers, specifically doctors, by providing concise and evidence-based clinical practice guidelines in the field of family medicine. The chatbot leverages Amazon Bedrock's anthropic.claude-instant-v1 model for natural language processing to retrieve relevant guidelines and generate responses tailored to the needs of healthcare professionals.

Note: This chatbot is intended for use by licensed healthcare providers only. It is currently under development and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Features
- Targeted Audience: Exclusively designed for licensed healthcare providers in the field of family medicine.
- Guideline Retrieval: Uses RAG techniques to fetch and present the most relevant clinical guidelines.
- Concise and Evidence-Based Responses: Generates responses that are clear, concise, and rooted in the latest evidence-based practices.
- Disclaimer Inclusion: Automatically includes a disclaimer with every response to clarify the intended use and limitations of the chatbot.

## View Demo

### 1. Click Code and create a Codespace on main.

### 2. Create account on AWS console
i. Log on to https://aws.amazon.com/console/.

ii. Click 'Sign in to the Console' on the top right corner. Sign up if you do not have an AWS account. 

iii. Create User Group.

iv. Create IAM user and assign Administrator Access to user. Be sure to download the AccessKey and save it in a folder on your local desktop.

v. Create EC2 instance. Configure instance security and make sure to add a new rule specifying port 8501.

vi. Create S3 bucket and a folder. Download files available in the 'Sample Files' section and upload it into the S3 folder. 

vii. Access Amazon Bedrock. Click 'Get Started'. Scroll down the left side bar, locate and click 'Model Access'. Request access for Anthropic Claude model.

### 3. Go back to Codespaces
Look for # Configure cached vector database in the ```chatbot_version1.py``` file.

Specify bucket_name and folder_path based on your S3 bucket and folder names.

Install all required libraries listed in the requirements.txt file

Type ```aws configure```

Key in your ACCESS KEY details. 

Type '''streamlit run chatbot_version1.py```

Open the 'External URL: xxxxxxxxxxx'

### 4. Playing with the chatbot
Once the chatbot is up and running, licensed healthcare providers can interact with it to retrieve clinical guidelines in family medicine. Simply ask a question or request guidance, and the chatbot will respond with relevant information, along with a standard disclaimer.

Example Interaction
User: What are the current guidelines for managing hypertension in adults?

Chatbot: According to the latest guidelines, first-line treatment for hypertension in adults includes lifestyle modifications such as diet and exercise, along with the use of antihypertensive medications like ACE inhibitors or ARBs. Please consult the full guidelines for more detailed recommendations.

This information is intended for use by licensed healthcare providers only and is not a substitute for professional medical advice, diagnosis, or treatment. This tool is currently under development, and while we strive for accuracy, there may be limitations in the information provided.

## Development Status
This project is currently under development. While the chatbot aims to provide accurate and up-to-date information, there may be limitations or inaccuracies. We are actively working on improving the system, and contributions are welcome.

## Contributing
We welcome contributions from the community. If you're interested in contributing, please follow these steps:

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
This project was inspired by the need for quick and reliable access to family medicine guidelines in clinical settings.
