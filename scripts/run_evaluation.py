from app.agent.agent import Agent
from app.agent.verification import CalculatorVerifier
from app.evaluation.cases import build_calculator_cases
from app.evaluation.evaluator import Evaluator
from app.llm.ollama import OllamaLLMClient


def main() -> None:
    llm = OllamaLLMClient()
    verifier = CalculatorVerifier()

    agent = Agent(
        llm_client=llm,
        verifier=verifier,
    )

    cases = build_calculator_cases()

    evaluator = Evaluator()

    result = evaluator.run(agent, cases, max_retries=1)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for case_result in result.case_results:
        status = "PASS" if case_result.passed else "FAIL"

        print(f"[{status}] {case_result.case_name}")
        print(f"    Reason: {case_result.verification_reason}")

        if case_result.error:
            print(f"    Error: {case_result.error}")

    print("=" * 60)
    print(f"Total cases:   {result.total_cases}")
    print(f"Passed cases:  {result.passed_cases}")
    print(f"Success rate:  {result.success_rate:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()