import pytest

from litchain.llm.llm_tool import camel_to_kebab


def test_light_chain():
    with pytest.raises(TypeError):
        camel_to_kebab()
