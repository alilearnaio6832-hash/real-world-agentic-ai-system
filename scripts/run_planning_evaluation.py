from app.agent.agent import Agent
from app.agent.planning import Planner
from app.llm.ollama import OllamaLLMClient
from app.tools.calculator import CalculatorTool
from app.tools.registry import ToolRegistry
from app.tools.text_analyzer import TextAnalyzerTool


def main() -> None:
    llm = OllamaLLMClient()
    planner = Planner(llm_client=llm)

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TextAnalyzerTool())

    agent = Agent(
        llm_client=llm,
        tool_registry=registry,
    )

    tasks = [
        "Calculate 25 * 4, then describe whether the result is large "
        "or small in one word.",
        "First calculate 10 + 5. Then count the words in the "
        "sentence 'The quick brown fox jumps'.",
    ]

    for task in tasks:
        print("=" * 60)
        print(f"TASK: {task}")
        print("=" * 60)

        result = agent.run_with_planning(
            task,
            tools=[],
            planner=planner,
        )

        print(f"Steps executed: {result.state.steps_executed}")
        print("Step results:")

        for index, step_result in enumerate(result.step_results, start=1):
            print(f"  {index}. {step_result!r}")

        print(f"Final answer: {result.content!r}")
        print()


if __name__ == "__main__":
    main()