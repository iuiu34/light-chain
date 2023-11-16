import pytest

from light_chain.llm.base_model import LlmBaseModel


def test_light_chain():
    with pytest.raises(TypeError):
        LlmBaseModel()


