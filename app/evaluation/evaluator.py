from dataclasses import dataclass, field

from app.agent.agent import Agent


@dataclass
class EvaluationCase:
    """A single evaluation scenario for the agent."""

    name: str
    task: str
    tools: list[dict]
    expects_tool_call: bool


@dataclass
class CaseResult:
    """The outcome of running one evaluation case."""

    case_name: str
    passed: bool
    used_tool_call: bool
    verification_reason: str
    error: str | None = None
    elapsed_seconds: float = 0.0
    total_tokens: int = 0


@dataclass
class EvaluationResult:
    """Aggregated results across all evaluation cases."""

    case_results: list[CaseResult] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.case_results)

    @property
    def passed_cases(self) -> int:
        return sum(
            1 for result in self.case_results if result.passed
        )

    @property
    def success_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0

        return self.passed_cases / self.total_cases

    @property
    def tool_selection_rate(self) -> float:
        """
        Fraction of cases that expected a tool call and got one,
        out of all cases that expected a tool call.
        """

        relevant = [
            result
            for result in self.case_results
            if result.used_tool_call is not None
        ]

        if not relevant:
            return 0.0

        correct = sum(
            1 for result in relevant if result.used_tool_call
        )

        return correct / len(relevant)

    @property
    def total_elapsed_seconds(self) -> float:
        """Sum of elapsed time across all cases."""

        return sum(
            result.elapsed_seconds for result in self.case_results
        )

    @property
    def total_tokens(self) -> int:
        """Sum of token usage across all cases."""

        return sum(
            result.total_tokens for result in self.case_results
        )


class Evaluator:
    """Runs a set of evaluation cases against an agent and measures results."""

    def run(
        self,
        agent: Agent,
        cases: list[EvaluationCase],
        max_retries: int = 1,
    ) -> EvaluationResult:
        """Execute every case and collect measurable results."""

        result = EvaluationResult()

        for case in cases:
            case_result = self._run_case(
                agent,
                case,
                max_retries,
            )

            result.case_results.append(case_result)

        return result

    def _run_case(
        self,
        agent: Agent,
        case: EvaluationCase,
        max_retries: int,
    ) -> CaseResult:
        """Run a single case and report whether it passed."""

        try:
            verified_result = agent.run_with_recovery(
                case.task,
                tools=case.tools,
                max_retries=max_retries,
            )

            used_tool_call = (
                case.expects_tool_call
                if not case.tools
                else None
            )

            return CaseResult(
                case_name=case.name,
                passed=verified_result.verification.passed,
                used_tool_call=used_tool_call,
                verification_reason=verified_result.verification.reason,
                elapsed_seconds=verified_result.state.elapsed_seconds,
                total_tokens=verified_result.state.total_tokens,
            )

        except Exception as exc:
            return CaseResult(
                case_name=case.name,
                passed=False,
                used_tool_call=None,
                verification_reason="Execution raised an exception.",
                error=str(exc),
            )