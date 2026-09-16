from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Represents the result of verifying an agent response."""

    passed: bool
    reason: str

    @property
    def status(self) -> str:
        """Return the verification status."""

        return "PASS" if self.passed else "FAIL"


class Verifier:
    """Base interface for verifying agent results."""

    def verify(
        self,
        task: str,
        result: str,
    ) -> VerificationResult:
        """
        Verify an agent result against the original task.

        Args:
            task: Original user task.
            result: Agent-generated result.

        Returns:
            VerificationResult containing PASS/FAIL status
            and an explanation.
        """

        raise NotImplementedError