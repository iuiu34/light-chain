# LitChain
![](https://img.shields.io/badge/version-v0.0.4-blue.svg)
![](https://img.shields.io/badge/python-3.9-blue.svg)
[![Docs](https://img.shields.io/badge/docs-confluence-013A97)]()
![](https://img.shields.io/badge/dev-orange.svg)

Simple wrapper around openai's client.chat.completions.create.
To simplify function/tool calling. [openai docs](https://platform.openai.com/docs/guides/function-calling)

This repo takes some inspiration from [langchain](https://python.langchain.com/docs/get_started/introduction.html). But does not use it. 

We think that, in [langchain](https://python.langchain.com/docs/get_started/introduction.html),
it's fast changes make it a good forum to be aware of the most recent advances and methods in llm usage.
But that is unstable, unnecessary complex, sometimes redundant, and makes you loss control over your code and prompts.
And that using a simple prompt and an external library is enough for a lot of cases.
So we don't want to use it in production.

## Usage
### Main operator

```python
from litchain import LlmBaseModel, tool


@tool
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location."""
    [...]


model = LlmBaseModel(
    system="GPT assistant",
    prompt_template="What's the weather in {city}?",
    tools=[get_current_weather],
    tool_choice="get_weather")

prompt = model.get_prompt(city="San Francisco")
model.predict_sample(prompt=prompt)
```

For full code, check examples/hello_world

### Examples
Run 
```sh
/examples/mail_classifier/mail_classifier.py --email "I want to buy a car"
```
for a simple example.

List of examples:
* hello_world.py: A simple example using get_current_weather func from openai docs.
* mail_classifier.py: A simple example of a mail classifier.
* movie_trivia_autogpt.py: A simple example of a movie trivia generator using sequential llm calls.

## Setup
### Instalation
```sh
pip install litchain
```

or from source
```sh
pip install -r requirements.txt
pip install -e .
```


### Env variables
You will need to set the following env variables:
* OPENAI_API_KEY: Your openai api key.



### Authors

*Iuiu* 


