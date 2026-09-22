from app.tools.base import Tool


class TextAnalyzerTool(Tool):
    """Analyzes basic statistics of a text: word and character counts."""

    @property
    def name(self) -> str:
        return "text_analyzer"

    @property
    def description(self) -> str:
        return (
            "Counts words and characters in a given text. "
            "Input: the text to analyze. "
            "Output: 'words: N, characters: M'."
        )

    def run(self, tool_input: str) -> str:
        if not tool_input.strip():
            raise ValueError("Input text cannot be empty.")

        word_count = len(tool_input.split())
        character_count = len(tool_input)

        return f"words: {word_count}, characters: {character_count}"