import json
import logging
import sys
import time

from openai import RateLimitError, OpenAI

from .messages import Messages

FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] %(name)-8s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.Logger(name)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(FORMATTER)
    logger.addHandler(stdout_handler)
    logger.setLevel(level)
    return logger


class LlmBaseModel:
    def __init__(self, model: str = "gpt-4-1106-preview", max_tokens: int = 500,
                 temperature: float = .2, top_p: int = 1,
                 prompt_template: str = None,
                 system: str = None,
                 tools: list = None,
                 tool_choice: dict = "auto",
                 eval_tools: bool = True,
                 reset_messages: bool = False,
                 verbose: bool = True,
                 ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reset_messages = reset_messages
        self.messages = Messages()
        self.eval_tools = eval_tools
        self.tools = tools
        self.verbose = verbose
        logger_level = logging.INFO if verbose else logging.WARNING
        logger_name = self.__class__.__name__
        self.logger = get_logger(logger_name, level=logger_level)
        if tool_choice != 'auto' and type(tool_choice) is str:
            tool_choice = {"type": "function", "function": {"name": tool_choice}}

        if tools is not None:
            self.tools_llm = self.get_tools_llm(tools)
            self.tool_choice = tool_choice

        # self.topics = get_topics()
        if system is None:
            system = self.get_system_default()

        self.system = system

        if prompt_template is None:
            prompt_template = self.get_prompt_template_default()

        self.prompt_template = prompt_template

    @staticmethod
    def get_system_default():
        out = \
            '''
You are an LLM API. Classifiy text with precision and return output in JSON format.'
        '''
        return out

    @staticmethod
    def get_prompt_template_default():
        out = \
            '''
            Classify prompt in one of the classes. 
            If you are not certain about the answer, you should return: "uncertain".
            Return output in JSON format

            Prompt:
            """
            {user_prompt}
            """

            Classes:
            """
            {topics}
            """
            '''  # noqa
        return out

    def get_prompt(self, prompt_template=None, **kwargs):
        if prompt_template is None:
            prompt_template = self.prompt_template
        prompt = prompt_template.format(**kwargs)
        return prompt

    @staticmethod
    def get_tools_llm(tools):
        return [v.__tool__ for v in tools]

    def parse_chat_completion_response(self, response):
        if response.tool_calls:
            for tool_call in response.tool_calls:

                func_name = tool_call.function.name
                kwargs = tool_call.function.arguments
                self.messages.add_assistant(kwargs,
                                            tool_calls=response.tool_calls)
                kwargs = json.loads(kwargs)
                self.logger.info(f'tool call: {func_name}(**{kwargs})')
                if self.eval_tools:
                    for tool in self.tools:
                        if tool.__name__ == func_name:
                            break
                    out = tool(**kwargs)
                    self.messages.add_tool(out, id=tool_call.id, name=func_name)
        else:
            out = response.content
            self.messages.add_assistant(out)

    def predict_sample(self, prompt=None, system=None, messages=None):
        if self.reset_messages and messages is not None:
            raise ValueError(f'reset_messages is {self.reset_messages} and memory is not None')

        if messages is not None:
            self.messages = messages

        elif self.reset_messages:
            self._reset_messages()

        if system is None:
            system = self.system

        self.messages.add_system(system)
        if prompt is not None:
            self.messages.add_user(prompt)
        out = None
        chat_args = dict(
            model=self.model,
            messages=self.messages(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p)
        if self.tools is not None:
            chat_args.update(
                tools=self.tools_llm,
                tool_choice=self.tool_choice)
        self.logger.info(f'message_history: {self.messages()}')

        while out is None:
            client = OpenAI()
            try:
                out = client.chat.completions.create(
                    **chat_args)
                out = out.choices[0].message
                self.parse_chat_completion_response(out)
            except RateLimitError:
                self.logger.warning('RateLimitError')
                self.logger.warning('sleeping for 10 seconds')
                time.sleep(10)

        self.logger.info(f'response: {self.messages.last_content()}')
        return self.messages

    def fit(self, x, y):
        self.columns = x.columns
        return self

    def predict(self, x):
        x_ = x.copy()
        x_.columns = [v.lower() for v in x_.columns]
        prompts = [self.get_prompt(**row) for _, row in x_.iterrows()]
        preds = [self.predict_sample(v) for v in prompts]
        return preds

    def _reset_messages(self):
        self.messages = Messages()
