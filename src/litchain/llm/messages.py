import json


class Messages:
    def __init__(self):
        self.messages = []

    def add_user(self, text):
        self.messages.append(
            dict(role='user', content=text)
        )

    def add_system(self, text):
        messages = [v for v in self.messages if v['role'] != 'system']
        self.messages = [dict(role='system', content=text)
                         ] + messages

    def add_assistant(self, text, tool_calls=None):
        message = dict(role='assistant', content=text)
        if tool_calls is not None:
            message['tool_calls'] = tool_calls

        self.messages.append(
            message
        )

    def add_tool(self, text, name, id):
        self.messages.append(
            dict(role='tool', content=text,
                 name=name, tool_call_id=id)
        )

    def __call__(self):
        return self.messages

    def last_content(self, n=-1):
        return self.messages[n]['content']

    def last_tool_name(self):
        for message in self.messages[::-1]:
            if message['role'] == 'tool':
                return message['name']

    def last_tool_args(self):
        n = 0
        for message in self.messages[::-1]:
            if n == 1:
                out = message['content']
                out = json.loads(out)
                return out

            if message['role'] == 'tool':
                n = 1
