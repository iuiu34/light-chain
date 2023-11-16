import importlib.resources as pkg

import fire

from litchain.llm.autogpt import AutoGPT
from litchain.llm.llm_tool import tool


def get_tools():
    from _utils.python_repl import python
    from _utils.wiki_search import search

    tools = [
        search,
        python
    ]
    tools = [tool(v) for v in tools]
    return tools


def get_prompt(filename: str = "system"):
    if "." not in filename:
        filename = f"{filename}.prompt"

    path = pkg.files('chainflow.movie_trivia')
    filename = path.joinpath('prompts', filename)
    with open(filename) as f:
        prompt = f.read()
    return prompt


def run_flow(film: str = "Lord of the Rings"):
    """Run the flow."""
    print('run_flow')
    print(vars())

    autogpt_args = dict(
        model="gpt-4",
        system=get_prompt("system"),
        # trigger_prompt=get_prompt("trigger"),
        prompt_template=get_prompt("prompt_template"),
        tools=get_tools(),
        n_iter=10
    )

    agent = AutoGPT(
        **autogpt_args
    )
    result = agent.predict(film=film)
    return result


def main():
    """Execute main program."""
    fire.Fire(run_flow)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
