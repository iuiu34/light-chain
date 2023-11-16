"""Main module."""
import importlib.resources as pkg

import pandas as pd
import yaml

from litchain.llm.base_model import LlmBaseModel
from litchain.llm.llm_tool import tool
import fire

def get_metadata():
    metadata = dict(
        booking_id="$booking_id",
        amount_and_currency="$amount_and_currency",
        provider_name="$provider_name",
        state_reason="$state_reason",
        refund_request_submission_date="$refund_request_submission_date"
    )

    return metadata


@tool
def get_email(label: str) -> str:
    """Get email.

    Args:
        label: label in the solution matrix to identify the email.
        """
    path = pkg.files('mail_classifer')
    filename = path.joinpath('prompts', 'solution_matrix.yaml')
    with open(filename, 'r') as f:
        emails = yaml.safe_load(f)
    labels = emails.keys()
    if label not in labels:
        template = None
        email_out = "Mail couldn't be classified in any of the categories."
    else:
        original_template = emails[label]["template"]
        template = emails[label]["template"]
        metadata = get_metadata()
        email_out = template.format(**metadata)
        template = original_template

    out = dict(label=label, email_out=email_out, template=template)
    return out


class LlmModel(LlmBaseModel):
    def __init__(self):
        path = pkg.files('mail_classifier')
        filename = path.joinpath('prompts', 'solution_matrix.yaml')
        with open(filename) as f:
            solution_matrix = yaml.safe_load(f)
        solution_matrix = [{"label": k} | v for k, v in solution_matrix.items()]
        solution_matrix = pd.DataFrame(solution_matrix)

        filename = path.joinpath('prompts', 'prompt_template.txt')
        with open(filename) as f:
            prompt_template = f.read()
        solution_matrix_ = solution_matrix[['label', 'scenario', 'scope']]
        prompt_template = prompt_template.format(
            solution_matrix=solution_matrix_.to_string(index=False)
        )

        filename = path.joinpath('prompts', 'system.txt')
        with open(filename) as f:
            system = f.read()

        super().__init__(
            system=system,
            prompt_template=prompt_template,
            tool_choice="get_email",
            tools=[get_email]
        )

def mail_classifier(email=None):
    if email is None:
        email = "I want to cancel my booking."
    model = LlmModel()
    prompt = model.get_prompt(email=email)
    p = model.predict_prompt(prompt)
    p = p.last_content()
    print(p)
    return p

def main():
    """Execute main program."""
    fire.Fire(mail_classifier)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
