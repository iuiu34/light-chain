import inspect

from pandas import DataFrame

from .base_model import LlmBaseModel
from .llm_tool import tool


@tool
def get_python_object(key: str, value: str) -> str:
    """Gets code for python object.

    Args:
        key: name of python object
        value: value of python object
        """
    return f"{key} = {value}"


class Bunch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self):
        out = {}
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.ismethod(i[1]):
                    out[i[0]] = i[1]
        return out

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"bunch has no attribute '{k}'")
            else:
                setattr(self, f"new_{k}", v)


class LlmVariable(LlmBaseModel):
    def __init__(self, value, key, bunch):
        super().__init__()
        self.original_type = type(value)
        self.size = len(value)
        self.type = type(value).__name__
        self.value = value

        if type(value) is DataFrame:
            value = value.to_dict(orient='list')
        self.value_ = value
        self.functions = [get_python_object]
        self.get_llm_tools()
        self.bunch_ = bunch
        self.reset_messages = True
        if "business_area" not in self.bunch_.keys():
            raise ValueError("bunch must have key 'business_area'")

        self.prompt_template_ = \
            '''
The python object bellow has type "{type_}", name "{key}", and size "{size}".
Contains values for the customer support area "{{business_area}}".

Give me a new python object for the customer support area "{{new_business_area}}".
Resulting python object should have same type, same name, and same size as input.

Return only python code

python_object (topics):
"""
{value}
"""
'''

        self.prompt_template = self.prompt_template_.format(
            value=str(self.value_).replace("{", "{{").replace("}", "}}"),
            type_=self.type,
            key=key,
            size=self.size)

        self.system = \
            """
You are an llm API. In the output return only python code.
Don't add comments to the code.
"""

    def __call__(self):
        new_business_area = self.bunch_['new_business_area']
        business_area = self.bunch_['business_area']

        if business_area != new_business_area:
            prompt = self.get_prompt(new_business_area=new_business_area,
                                     business_area=business_area)
            p = self.predict_sample(prompt)
            p = p.last_function_args()
            p = p['value']
            p = eval(p)
            if self.original_type is DataFrame:
                p = DataFrame(p)
            self.value = p
        return self.value

    def get_bunch(self):
        return self.bunch()

# topics = ["shoe care", "shoe care"]
# bunch = Bunch(business_area='a')
# topics_ = LlmVariable(topics, 'topics', bunch=bunch)
# print(topics_())
# bunch.update(business_area='b')
# # topics_ = LlmVariable(topics)
# print(topics_())
