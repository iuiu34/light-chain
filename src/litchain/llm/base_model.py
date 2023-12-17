import json
import logging
import os
import sys
import time

from joblib import Parallel, delayed
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
    def __init__(self, model: str = "gpt-4-1106-preview", max_tokens: int = 1500,
                 temperature: float = .2, top_p: int = 1,
                 prompt_template: str = None,
                 system: str = None,
                 tools: list = None,
                 tool_choice: dict = "auto",
                 eval_tools: bool = True,
                 reset_messages: bool = True,
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
        ChatGPT API.
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

    def get_dependencies(self, tool, messages):
        dependencies = tool.__tool_dependencies__
        name = tool.__name__
        previous_tools_response = messages.tools_response()

        kwargs_ = {}
        for k, v in dependencies.items():
            if type(v) is str:
                kwargs_[k] = previous_tools_response[v]
            elif type(v) in [list, set] and len(v) == 2:
                v, w = v
                kwargs_[k] = previous_tools_response[v][w]
            else:
                raise ValueError(f"tool {name}. dependencies {k} {v}")
        return kwargs_

    def _eval_tools(self, func_name, tool_call, kwargs, messages):
        for tool in self.tools:
            if tool.__name__ == func_name:
                break
        if tool.__tool_dependencies__:
            kwargs_ = self.get_dependencies(tool, messages)
            kwargs.update(kwargs_)
        out = tool(**kwargs)
        messages.add_tool(out, id=tool_call.id, name=func_name)
        return messages

    def parse_chat_completion_response(self, response, messages):
        if response.tool_calls:
            for tool_call in response.tool_calls:
                func_name = tool_call.function.name
                kwargs = tool_call.function.arguments
                messages.add_assistant(kwargs,
                                       tool_calls=response.tool_calls)
                self.logger.info(f'tool call: {func_name}(**{kwargs})')
                if type(kwargs) is str:
                    kwargs = json.loads(kwargs)
                # self.logger.info(f'tool call: {func_name}(**{kwargs})')
                if self.eval_tools:
                    messages = self._eval_tools(func_name, tool_call, kwargs, messages)
        else:
            out = response.content
            messages.add_assistant(out)
        return messages

    def predict_sample_gpt(self):
        pass

    def predict_sample_gemini(self):
        import vertexai
        from vertexai.preview import generative_models
        from vertexai.preview.generative_models import GenerativeModel

        # vertexai.init(project='ds-mkt', location='us-central1')
        vertexai.init(project=self.project, location=self.location)

        model = GenerativeModel("gemini-pro")
        get_current_weather_func = None
        weather_tool = generative_models.Tool(
            function_declarations=[get_current_weather_func],
        )

        chat = model.start_chat()
        model_response = chat.send_message(
            "What is the weather like in Boston?",
            tools=[weather_tool])
        return model_response

    def predict_sample(self, prompt=None, system=None, messages=None):
        if messages is None:
            messages = Messages()

        if system is None:
            system = self.system

        messages.add_system(system)
        if prompt is not None:
            prompt_ = prompt.replace('\n', '\\n')
            self.logger.info(f"prompt: {prompt_}")
            messages.add_user(prompt)
        out = None
        chat_args = dict(
            model=self.model,
            messages=messages(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p)
        if self.tools is not None:
            chat_args.update(
                tools=self.tools_llm,
                tool_choice=self.tool_choice)
        self.logger.info(f'message_history: {messages()}')

        while out is None:
            client = OpenAI()
            try:
                out = client.chat.completions.create(
                    **chat_args)
                out = out.choices[0].message
                messages = self.parse_chat_completion_response(out, messages)
            except RateLimitError:
                self.logger.warning('RateLimitError')
                self.logger.warning('sleeping for 10 seconds')
                time.sleep(10)

        self.logger.info(f'response: {messages.last_content()}')
        return messages

    def fit(self, x, y):
        self.columns = x.columns
        return self

    def predict(self, x, parallel=False):
        x_ = x.copy()
        x_.columns = [v.lower() for v in x_.columns]
        prompts = [self.get_prompt(**row) for _, row in x_.iterrows()]
        if self.verbose:
            self.logger.info(f"pretty prompt:\n{prompts[0]}")

        def process(x):
            return self.predict_sample(x)

        if parallel:
            print("parallel")
            preds = Parallel(n_jobs=-1)(
                delayed(process)(v)
                for v in prompts)
        else:
            preds = [
                process(v)
                for v in prompts]
        return preds

    def _reset_messages(self):
        self.messages = Messages()

    def app_(self):
        import streamlit as st
        st.subheader("Config")
        self.system = st.text_area("system", self.system)
        self.prompt_template = st.text_area("prompt_template", self.prompt_template)
        self.reset_messages = st.checkbox("reset_messages", self.reset_messages)

        prompt = st.chat_input()
        st.subheader("messages")
        if prompt:
            self.predict_sample(prompt)

        for message in self.messages():
            with st.chat_message(message["role"]):
                st.text(message["content"])

    def app(self, package_path=None):
        from streamlit import config as _config
        from streamlit.web.bootstrap import run
        name = self.__class__.__name__
        if package_path is None and name != "LlmBaseModel":
            raise ValueError("package_path must be provided if model is not LlmBaseModel")
        if package_path is None:
            package_path = "edo.mkt.ml.llm"
        app = f"""from {package_path} import {name};model={name}();model.app_()"""
        # app = f"import streamlit as st;st.text('test')"

        tmp_file_path = os.path.join("tmp", 'app.py')
        with open(tmp_file_path, 'w') as f:
            f.write(app)

        _config.set_option("server.headless", True)

        run(tmp_file_path, '', [], [])
