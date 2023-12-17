import json


class Messages:
    def __init__(self):
        self.messages = []

    def add_user(self, content):
        self.messages.append(
            dict(role='user', content=content)
        )

    def add_system(self, content):
        messages = [v for v in self.messages if v['role'] != 'system']
        self.messages = [dict(role='system', content=content)
                         ] + messages

    def add_assistant(self, content, tool_calls=None):
        message = dict(role='assistant', content=content)
        if tool_calls is not None:
            message['tool_calls'] = tool_calls

        self.messages.append(
            message
        )

    def add_tool(self, content, name, id):
        self.add_tool_response(content, name, id)

        if type(content) is dict:
            try:
                content = json.dumps(content)
            except:
                pass

        self.messages.append(
            dict(role='tool', content=str(content),
                 name=name, tool_call_id=id)
        )

    def add_tool_response(self, content, name, id=None):
        self.messages.append(
            dict(role='tool_response', content=content,
                 name=name, tool_call_id=id)
        )

    def __call__(self, response=False):
        messages = self.messages
        if not response:
            messages = [v for v in self.messages if v['role'] != 'tool_response']
        return messages

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

    def last_tool_response(self):
        for message in self.messages[::-1]:
            if message['role'] == 'tool_response':
                out = message['content']
                return out

    def tools_response(self):
        out = {}
        for message in self.messages:
            if message['role'] == 'tool_response':
                out[message['name']] = message['content']
        return out
