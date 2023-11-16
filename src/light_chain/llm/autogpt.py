from .base_model import LlmBaseModel
from .llm_tool import tool


@tool
def finish(response: str) -> str:
    """
    Use this to signal that you have finished all your goals and remember show your results.

   Args:
       response: final response to let people know you have finished your goals and remember show your results.
    """
    return response


class AutoGPT(LlmBaseModel):
    def __init__(
            self,
            system,
            prompt_template,
            trigger=None,
            tools=None,
            reset_messages=False,
            model: str = "gpt-4-1106-preview",
            n_iter: int = 10,
            verbose: bool = True,
    ):

        if trigger is None:
            trigger = self.get_trigger_default()
        if system is None:
            system = self.get_system_default()
        tools.append(finish)
        self.n_iter = n_iter
        self.trigger = trigger

        super().__init__(model=model,
                         system=system,
                         tools=tools,
                         prompt_template=prompt_template,
                         reset_messages=reset_messages,
                         verbose=verbose)

    @staticmethod
    def get_trigger_default():
        return "Determine which next function to use, and respond using stringfield JSON object.\nIf you have completed all your tasks, make sure to use the 'finish' function to signal and remember show your results."

    @staticmethod
    def get_system_default():
        return "You are an LLM API."

    def predict_sample(self, **kwargs):
        for n in range(self.n_iter):
            self.logger.info(f"n_iter: {n}")
            if n == 0:
                prompt = self.get_prompt(**kwargs)
            else:
                prompt = self.trigger

            messages = super().predict_sample(prompt)
            tool_name = messages.last_tool_name()
            if tool_name == "finish":
                return messages

        raise ValueError(f"Could not finish in {self.n_iter} iterations")

    def predict(self, **kwargs):
        response = self.predict_sample(**kwargs)
        return response
