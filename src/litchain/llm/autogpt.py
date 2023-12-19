from .base_model import LlmBaseModel
from .llm_tool import tool

from .messages import Messages


@tool
def finish():
    """
    Call this function to signal that you have finished all your goals.

    """
    print("finish")
    print(vars())


class AutoGPT(LlmBaseModel):
    def __init__(
            self,
            system,
            prompt_template,
            trigger=None,
            tools=None,
            last_tool_name=None,
            reset_messages=False,
            model: str = "gpt-4-1106-preview",
            n_iter: int = 10,
            verbose: bool = True,
            st_empty=None,
    ):

        # if trigger is None:
        #     trigger = self.get_trigger_default()
        if system is None:
            system = self.get_system_default()
        if last_tool_name is None:
            last_tool_name = "finish"
        tools.append(finish)
        self.n_iter = n_iter
        self.trigger = trigger
        self.last_tool_name = last_tool_name
        self.st_empty = st_empty
        super().__init__(model=model,
                         system=system,
                         tools=tools,
                         prompt_template=prompt_template,
                         reset_messages=reset_messages,
                         verbose=verbose
                         )

    @staticmethod
    def get_trigger_default():
        out = \
            """
            Determine which next function to use, and respond using stringfield JSON object.
            If you have completed all your tasks, make sure to use the 'finish' function"
            to signal and remember show your results.
            """
        return out

    @staticmethod
    def get_system_default():
        return "You are an LLM API."

    def get_st_empty(self, tool_name, tool_content):
        st_log = (
            ":bulb:\n\n"
            f"tool name: {tool_name}()\n\n"
            f"tool content: {tool_content}\n\n")
        self.st_empty.markdown(st_log)

    def predict_sample(self, prompt=None, system=None, messages=None):
        if messages is None:
            messages = Messages()
        for n in range(self.n_iter):

            self.logger.info(f"n_iter: {n}")
            if n != 0:
                prompt = self.trigger

            messages = super().predict_sample(
                prompt=prompt, system=system, messages=messages)

            tool_name = messages.last_tool_name()
            tool_content = messages.last_content()
            if self.st_empty is not None:
                self.get_st_empty(tool_name, tool_content)

            if tool_name == self.last_tool_name:
                return messages

        raise ValueError(f"Could not finish in {self.n_iter} iterations")

    # def predict(self, **kwargs):
    #     response = self.predict_sample(**kwargs)
    #     return response
