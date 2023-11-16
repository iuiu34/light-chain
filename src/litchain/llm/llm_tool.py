import inspect
from typing import Callable, List, Tuple


def camel_to_kebab(x):
    out = [x[0].lower()]
    for char in x[1:]:
        if char.isupper():
            out.append('_')
        out.append(char.lower())
    return ''.join(out)


def convert_to_openai_function(function, name=None):
    schema = function.schema()
    if name is None:
        name = schema["title"]
        name = camel_to_kebab(name)

    out = {
        "name": name,
        "description": schema["description"],
        "parameters": schema,
    }
    return out


class OpenaiTool:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
        the docstring has Google Python style argument descriptions, these will be
        included as well.
    """

    def __init__(self, function):
        description, arg_descriptions = self._parse_python_tool_docstring(function)
        output = self._parse_python_tool_output(function)
        if output != 'str':
            raise ValueError(f'Openai function output must be str. Not {output}')
        self.PYTHON_TO_JSON_TYPES = {
            "str": "string",
            "int": "number",
            "float": "number",
            "bool": "boolean",
        }
        self.openai_function = {
            "type": "function",
            "function": {
                "name": self._get_python_tool_name(function),
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": self._get_python_tool_arguments(function, arg_descriptions),
                    "required": self._get_python_tool_required_args(function)
                }
            }
        }

    def __call__(self):
        return self.openai_function

    @staticmethod
    def _parse_python_tool_docstring(function: Callable) -> Tuple[str, dict]:
        """Parse the function and argument descriptions from the docstring of a function.

        Assumes the function docstring follows Google Python style guide.
        """
        docstring = inspect.getdoc(function)
        if docstring:
            docstring_blocks = docstring.split("\n\n")
            descriptors = []
            args_block = None
            past_descriptors = False
            for block in docstring_blocks:
                if block.startswith("Args:"):
                    args_block = block
                    break
                elif block.startswith("Returns:") or block.startswith("Example:"):
                    # Don't break in case Args come after
                    past_descriptors = True
                elif not past_descriptors:
                    descriptors.append(block)
                else:
                    continue
            description = " ".join(descriptors)
        else:
            description = ""
            args_block = None
        arg_descriptions = {}
        if args_block:
            arg = None
            for line in args_block.split("\n")[1:]:
                if ":" in line:
                    arg, desc = line.split(":")
                    arg_descriptions[arg.strip()] = desc.strip()
                elif arg:
                    arg_descriptions[arg.strip()] += " " + line.strip()
        return description, arg_descriptions

    @staticmethod
    def _get_python_tool_name(function: Callable) -> str:
        """Get the name of a Python function."""
        source = inspect.getsource(function)
        source = source.split("def ")[1]
        source = source.split("(")[0]
        return source

    def _get_python_tool_arguments(self, function: Callable, arg_descriptions: dict) -> dict:
        """Get JsonSchema describing a Python functions arguments.

        Assumes all function arguments are of primitive types (int, float, str, bool) or
        are subclasses of pydantic.BaseModel.
        """
        properties = {}
        annotations = inspect.getfullargspec(function).annotations
        for arg, arg_type in annotations.items():
            if arg == "return":
                continue
            # if isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            #     properties[arg] = arg_type.schema()
            if arg_type.__name__ in self.PYTHON_TO_JSON_TYPES:
                properties[arg] = {"type": self.PYTHON_TO_JSON_TYPES[arg_type.__name__]}
            if arg in arg_descriptions:
                if arg not in properties:
                    properties[arg] = {}
                properties[arg]["description"] = arg_descriptions[arg]
        return properties

    @staticmethod
    def _get_python_tool_required_args(function: Callable) -> List[str]:
        """Get the required arguments for a Python function."""
        spec = inspect.getfullargspec(function)
        required = spec.args[: -len(spec.defaults)] if spec.defaults else spec.args
        required += [k for k in spec.kwonlyargs if k not in (spec.kwonlydefaults or {})]
        return required

    @staticmethod
    def _parse_python_tool_output(function: Callable) -> str:

        return inspect.getfullargspec(function).annotations['return'].__name__


def tool(function):
    """Assumes function docs in google style.
      All function arguments are of primitive types (int, float, str, bool) or
        Output is str."""
    openai_tool_ = OpenaiTool(function)
    function.__tool__ = openai_tool_()
    return function
