import json

from promptflow import tool

from autogpt_class import generate_function


# @tool

def functions_format() -> list:
    functions = [
        {
            "name": "search",
            "description": """The action will search this entity name on Wikipedia and returns the first {count}
            sentences if it exists. If not, it will return some related entities to search next.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name which is used for Wikipedia search.",
                    },
                    "count": {
                        "type": "integer",
                        "default": 10,
                        "description": "Returned sentences count if entity name exists Wikipedia.",
                    },
                },
                "required": ["entity"],
            },
        },
        {
            "name": "python",
            "description": """A Python shell. Use this to execute python commands. Input should be a valid python
            command and you should print result with `print(...)` to see the output.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command you want to execute in python",
                    }
                },
                "required": ["command"]
            },
        },
        {
            "name": "finish",
            "description": """use this to signal that you have finished all your goals and remember show your
            results""",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "final response to let people know you have finished your goals and remember "
                                       "show your results",
                    },
                },
                "required": ["response"],
            },
        },
    ]
    return functions


out = functions_format()
print(out)
with open("../../../../../tmp/test.json", "w") as f:
    json.dump(out, f, indent=4)

from wiki_search import search as search_
from python_repl import python


def finish(response: str) -> str:
    """
    Use this to signal that you have finished all your goals and remember show your\n            results",
    """
    return response


def search(entity: str, count: int = 10):
    """
    The input is an exact entity name. The action will search this entity name on Wikipedia and returns the first
    count sentences if it exists. If not, it will return some related entities to search next.

    Args:
        entity: Entity name which is used for Wikipedia search.
        count: Returned sentences count if entity name exists Wikipedia.
    """
    search_(**locals())


tools = [
    search,
    python,
    tool
]
tools += [finish]
functions = [generate_function(tool) for tool in tools]
print(functions)
with open("../../../../../tmp/test2.json", "w") as f:
    json.dump(functions, f, indent=4)
