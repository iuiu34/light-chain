"""Utils."""
import json
import os
import warnings

import fire
from cryptography.fernet import Fernet


def get_secret(secret_name: str, prefix: str = None,
               secret_client=None) -> str:
    out = os.getenv(secret_name)

    if out:
        return out

    warnings.warn(
        f"Getting {secret_name} from gcloud secrets.\nDeclaring {secret_name} as an ENV var has faster access.")

    from google.cloud import secretmanager
    if secret_client is None:
        secret_client = secretmanager.SecretManagerServiceClient()
    # if prefix is None:
    #     prefix = 'ds-dev'

    secret_name_ = secret_name.lower().replace('_', '-')
    project = os.environ["PROJECT"]
    if prefix is not None:
        prefix_ = prefix.lower().replace('_', '-')
        secret_name_ = f"{prefix_}-{secret_name_}"

    secret_name_ = f'projects/{project}/secrets/{secret_name_}/versions/latest'
    out = secret_client.access_secret_version(request={"name": secret_name_})
    out = out.payload.data.decode("UTF-8")

    os.environ[secret_name] = out
    return out


def save_secrets(variables, filename, key):
    from google.cloud import secretmanager

    secret_client = secretmanager.SecretManagerServiceClient()
    secrets = {}
    for var in variables:
        secret = get_secret(var, secret_client=secret_client)
        secrets[var] = secret
        os.environ[var] = secret

    secrets_txt = json.dumps(secrets)
    secrets_enc = Fernet(key.encode()).encrypt(secrets_txt.encode())
    with open(filename, 'w') as f:
        f.write(secrets_enc.decode())


def load_secrets(variables, filename, key):
    with open(filename) as f:
        secrets_enc = f.read()

    secrets_enc = Fernet(key.encode()).decrypt(secrets_enc.encode())
    secrets = json.loads(secrets_enc.decode())
    for var in variables:
        os.environ[var] = secrets[var]


def get_secrets(variables: list = None):
    key = os.getenv('GENERIC_KEY', '0T35mH3E4-RXW-c6_AgSaa6RM8ik1-T1yR7zKoD71Jw=')
    if variables is None:
        variables = ['OPENAI_API_KEY']
    filename = 'tmp/secrets.txt'
    # key = os.environ['GENERIC_KEY']

    if os.path.exists(filename):
        load_secrets(variables, filename, key)

    else:
        save_secrets(variables, filename, key)


def main():
    """Execute main program."""
    fire.Fire(get_secrets)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
