import json
import warnings

from . import LlmBaseModel


class GeminiBaseModel(LlmBaseModel):
    def __init__(self,
                 model: str = "gemini-pro",
                 system: str = None,
                 project: str = None,
                 location: str = 'us-central1',
                 **kwargs
                 ):
        if not model.startswith('gemini'):
            raise ValueError(f"model {self.model} not supported")

        self.project = project
        if location != 'us-central1':
            raise ValueError(f"You are using model {model}. Location {location} is not supported.")
        self.location = location

        if system is not None:
            warnings.warn("System appended to prompt. Gemini doesn't accept systems.")

        super().__init__(
            model=model,
            system=system,
            **kwargs
        )

    @staticmethod
    def _hasattr(obj, attr):
        try:
            getattr(obj, attr)
            out = True
        except ValueError:
            out = False
        return out

    def _parse_message(self, response, messages):
        part = response.candidates[0].content.parts[0]
        if self._hasattr(part, 'text'):
            out = part.text
            messages.add_assistant(out)

        elif self._hasattr(part, 'function_call'):
            tool_call = part.function_call
            kwargs = tool_call.args
            kwargs = {k: v for k, v in kwargs.items()}
            kwargs = json.dumps(kwargs)
            #     for tool_call in response.tool_calls:
            func_name = tool_call.name
            messages.add_assistant(kwargs,
                                   tool_calls=[func_name])
            self.logger.info(f'tool call: {func_name}(**{kwargs})')
            if type(kwargs) is str:
                kwargs = json.loads(kwargs)
            if self.eval_tools:
                messages = self._eval_tools(func_name, tool_call, kwargs, messages,
                                            id='gemini')

        return messages

    def _get_chat_response(self, messages):
        import vertexai
        from vertexai.preview import generative_models
        from vertexai.preview.generative_models import GenerativeModel, Part

        vertexai.init(project=self.project, location=self.location)

        generation_config = dict(
            temperature=self.temperature,
            top_p=self.top_p,
            # top_k=20,
            # candidate_count=1,
            max_output_tokens=self.max_tokens,
            # stop_sequences=["STOP!"],
        )
        if self.tools is not None:
            tools_declaration = [
                generative_models.FunctionDeclaration(
                    **tool['function'])
                for tool in self.tools_llm]
            tools_gemini = generative_models.Tool(
                function_declarations=[tool_declaration
                                       for tool_declaration in tools_declaration])
            tools_gemini = [tools_gemini]
        else:
            tools_gemini = None

        model = GenerativeModel(
            self.model,
            generation_config=generation_config,
            tools=tools_gemini)

        if not hasattr(messages, 'chat'):
            # todo add system
            chat = model.start_chat()
            chat.send_message(messages.system)
            # print(out)
            messages.chat = chat

        chat = messages.chat

        # chat.send_message(self.system)

        if messages.last_role() == 'user':
            prompt = messages.last_content()
        elif messages.last_role() == 'tool':
            tool_response = messages.last_tool_response()
            tool_name = messages.last_tool_name()
            prompt = Part.from_function_response(
                name=tool_name,
                response={
                    "content": tool_response,
                }
            )
        else:
            raise ValueError(f"last role {messages.last_role()} not supported")

        out = chat.send_message(prompt)
        messages = self._parse_message(out, messages)

        return messages
