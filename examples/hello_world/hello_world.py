import json

import fire

from light_chain import LlmBaseModel, tool


@tool
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location.

    Args:
        location (str): The city and state, e.g. San Francisco, CA.
        unit (str): Unit of temperature. Value should be in ["celsius", "fahrenheit"].
        """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def hello_world():
    model = LlmBaseModel(
        system="GPT assistant",
        prompt_template="What's the weather in {city}?",
        tools=[get_current_weather],
        tool_choice="get_weather")

    prompt = model.get_prompt(city="San Francisco")
    p = model.predict_sample(prompt=prompt)
    p = p.last_content()
    print(p)


def main():
    """Execute main program."""
    fire.Fire(hello_world)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
