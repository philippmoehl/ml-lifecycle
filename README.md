Plant Health
=================

![alt text](imgs/streamlit.gif "gif")

Description
-----------------
Predict plant health based on an image using deep learning. With the help of OpenAi's chatGPT you can directly generate a step-by-step solution in the web applicaiton. App is availbale at [philippmoehl-plant-disease-app-zgrf0b.streamlit.app/](https://philippmoehl-plant-disease-app-zgrf0b.streamlit.app/). If you want to run the app locally, or develop on top of it refer to the next sections. Please note that the deployed app does not have the chatGPT feature as you will need to use your credentials.


Installation
---------------

### Setup
1. Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)

2. Clone the repository.

3. Install the necessary dependencies:

`> pip install -r requirements.txt`


### Configuration

1. Find the file named `.env.template` in the main folder. This file may
    be hidden by default in some operating systems due to the dot prefix.
2. Create a copy of `.env.template` and call it `.env`;
    if you're already in a command prompt/terminal window: 
    
`> cp .env.template .env`.

3. Open the `.env` file in a text editor.
4. Find the line that says `OPENAI_API_KEY=`.
5. After the `=`, enter your unique OpenAI API Key *without any quotes or spaces*.
6. Save and close the `.env` file.


Usage
-----------------

### Configurations

1. Adapt the prompt design for your needs. The prompt template for solutions is located at `src/prompt_design.txt`. 
Sources for best practices in prompt engineering are for example [reddit](https://www.reddit.com/r/PromptEngineering/) 
and [openai docs](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api).
2. Adapt the `config.yaml` file with openai related configurations. Currently, it is [best 
practice](https://platform.openai.com/docs/guides/gpt) to use the "gpt-3.5-turbo" model, because of the cost and 
performance. If an increase in performance is required, "gpt-4" model can also be set. Note that
the [pricing](https://openai.com/pricing) of "gpt-4" is by far higher.

### Run
Web application built with [streamlit.io](https://streamlit.io/). To run locally:

`> streamlit run app.py`

Disclaimer
---------------
Please note that the use of a GPT language model and text-to-image models can be expensive due to its token usage. By utilizing this project, 
you acknowledge that you are responsible for monitoring and managing your own token usage and the associated costs. It 
is highly recommended to check your OpenAI API usage regularly and set up any necessary limits or alerts to prevent unexpected charges.