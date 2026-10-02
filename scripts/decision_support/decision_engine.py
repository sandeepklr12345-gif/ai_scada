from dataclasses import dataclass
from typing import Optional


@dataclass
class DecisionInput:
    forecast_2min: Optional[float] = None
    forecast_10min: Optional[float] = None
    forecast_30min: Optional[float] = None

    anomaly_score: Optional[float] = None
    anomaly_state: str = "INSUFFICIENT_HISTORY"
    anomaly_unavailable_reason: Optional[str] = None

    persistent_anomaly_count: int = 0

    predicted_attack_mechanism: Optional[str] = None
    classifier_score: Optional[float] = None


@dataclass
class DecisionResult:
    decision_level: str
    action: str
    reason: str
    predicted_attack_mechanism: Optional[str] = None
    classifier_score: Optional[float] = None
    anomaly_score: Optional[float] = None


class DecisionSupportEngine:

    def __init__(self, persistent_anomaly_threshold: int = 3):
        self.persistent_anomaly_threshold = (
            persistent_anomaly_threshold
        )

    def evaluate(
        self,
        decision_input: DecisionInput,
    ) -> DecisionResult:

        state = decision_input.anomaly_state.upper()

        # --------------------------------------------------------
        # Level 3
        # --------------------------------------------------------

        if (
            decision_input.persistent_anomaly_count
            >= self.persistent_anomaly_threshold
        ):
            return DecisionResult(
                decision_level="LEVEL_3",
                action="HUMAN_ONLY",
                reason=(
                    "Persistent anomaly condition reached the "
                    "configured escalation threshold."
                ),
                predicted_attack_mechanism=(
                    decision_input.predicted_attack_mechanism
                ),
                classifier_score=(
                    decision_input.classifier_score
                ),
                anomaly_score=(
                    decision_input.anomaly_score
                ),
            )

        # --------------------------------------------------------
        # Level 2
        # --------------------------------------------------------

        if state == "ANOMALY":
            return DecisionResult(
                decision_level="LEVEL_2",
                action="APPROVAL_REQUIRED",
                reason=(
                    "Anomaly detected by the HAI Candidate C "
                    "detector."
                ),
                predicted_attack_mechanism=(
                    decision_input.predicted_attack_mechanism
                ),
                classifier_score=(
                    decision_input.classifier_score
                ),
                anomaly_score=(
                    decision_input.anomaly_score
                ),
            )

        if state in {"INSUFFICIENT_HISTORY", "UNAVAILABLE"}:
            reason = decision_input.anomaly_unavailable_reason
            if reason is None:
                if state == "UNAVAILABLE":
                    reason = (
                        "Anomaly assessment is unavailable because "
                        "no compatible detector is available for this "
                        "feature schema."
                    )
                else:
                    reason = (
                        "Anomaly assessment is unavailable because "
                        "sufficient streaming history is not available."
                    )

            return DecisionResult(
                decision_level="LEVEL_2",
                action="APPROVAL_REQUIRED",
                reason=reason,
                predicted_attack_mechanism=(
                    decision_input.predicted_attack_mechanism
                ),
                classifier_score=(
                    decision_input.classifier_score
                ),
                anomaly_score=(
                    decision_input.anomaly_score
                ),
            )

        # --------------------------------------------------------
        # Level 1
        # --------------------------------------------------------

        if state == "NORMAL":
            return DecisionResult(
                decision_level="LEVEL_1",
                action="AUTOMATIC",
                reason="No anomaly detected.",
                predicted_attack_mechanism=(
                    decision_input.predicted_attack_mechanism
                ),
                classifier_score=(
                    decision_input.classifier_score
                ),
                anomaly_score=(
                    decision_input.anomaly_score
                ),
            )

        # --------------------------------------------------------
        # No UNKNOWN state
        # --------------------------------------------------------

        raise ValueError(
            f"Unsupported anomaly state: "
            f"'{decision_input.anomaly_state}'. "
            "Expected NORMAL, ANOMALY, or "
            "INSUFFICIENT_HISTORY or UNAVAILABLE."
        )


if __name__ == "__main__":

    engine = DecisionSupportEngine()

    test_cases = [
        (
            "NORMAL",
            DecisionInput(
                forecast_2min=350.0,
                forecast_10min=345.0,
                forecast_30min=340.0,
                anomaly_score=0.09,
                anomaly_state="NORMAL",
                predicted_attack_mechanism=None,
                classifier_score=None,
            ),
        ),
        (
            "ANOMALY_WITH_CLASSIFIER",
            DecisionInput(
                forecast_2min=350.0,
                forecast_10min=345.0,
                forecast_30min=340.0,
                anomaly_score=-0.05,
                anomaly_state="ANOMALY",
                predicted_attack_mechanism="AP14",
                classifier_score=0.82,
            ),
        ),
        (
            "INSUFFICIENT_HISTORY",
            DecisionInput(
                anomaly_score=None,
                anomaly_state="INSUFFICIENT_HISTORY",
            ),
        ),
        (
            "PERSISTENT_ANOMALY",
            DecisionInput(
                forecast_2min=350.0,
                forecast_10min=345.0,
                forecast_30min=340.0,
                anomaly_score=-0.08,
                anomaly_state="ANOMALY",
                persistent_anomaly_count=3,
                predicted_attack_mechanism="AP14",
                classifier_score=0.91,
            ),
        ),
    ]

    print("=" * 70)
    print("DECISION SUPPORT ENGINE SELF TEST")
    print("=" * 70)

    for name, decision_input in test_cases:

        result = engine.evaluate(
            decision_input
        )

        print(f"\n{name}")
        print("-" * 70)
        print(
            f"Decision level : "
            f"{result.decision_level}"
        )
        print(
            f"Action         : "
            f"{result.action}"
        )
        print(
            f"Reason         : "
            f"{result.reason}"
        )
        print(
            f"Attack         : "
            f"{result.predicted_attack_mechanism}"
        )
        print(
            f"Classifier     : "
            f"{result.classifier_score}"
        )
        print(
            f"Anomaly score  : "
            f"{result.anomaly_score}"
        )

    print("\n" + "=" * 70)
    print("DECISION SUPPORT ENGINE SELF TEST: PASS")
    print("=" * 70)
