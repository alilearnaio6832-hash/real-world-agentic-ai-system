import pytest

from app.tools.text_analyzer import TextAnalyzerTool


def test_text_analyzer_counts_words_and_characters():
    tool = TextAnalyzerTool()

    result = tool.run("Hello world foo")

    assert result == "words: 3, characters: 15"


def test_text_analyzer_single_word():
    tool = TextAnalyzerTool()

    result = tool.run("Hello")

    assert result == "words: 1, characters: 5"


def test_text_analyzer_rejects_empty_input():
    tool = TextAnalyzerTool()

    with pytest.raises(
        ValueError,
        match="Input text cannot be empty",
    ):
        tool.run("")


def test_text_analyzer_rejects_whitespace_only_input():
    tool = TextAnalyzerTool()

    with pytest.raises(
        ValueError,
        match="Input text cannot be empty",
    ):
        tool.run("   ")


def test_text_analyzer_has_correct_name():
    tool = TextAnalyzerTool()

    assert tool.name == "text_analyzer"