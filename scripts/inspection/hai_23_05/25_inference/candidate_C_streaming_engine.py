import pandas as pd

from candidate_C_inference_engine import (
    CandidateCInferenceEngine
)


class CandidateCStreamingEngine:
    """
    Stateful streaming wrapper around Candidate C.

    Maintains the previous 4 SCADA observations so that
    temporal features remain continuous across incoming
    chunks.

    Candidate C requires:
        - 1 previous observation for abs_diff_1s
        - 5 observations for rolling_std_5s

    Therefore, four previous rows are retained between
    chunks.
    """

    HISTORY_SIZE = 4

    def __init__(
        self,
        model_path,
        manifest_path
    ):

        self.engine = CandidateCInferenceEngine(
            model_path,
            manifest_path
        )

        self.history = None

    # ========================================================
    # PROCESS ONE CHUNK
    # ========================================================

    def predict_chunk(self, chunk):

        chunk = chunk.copy()

        # ----------------------------------------------------
        # First chunk
        # ----------------------------------------------------

        if self.history is None:

            combined = chunk.copy()

        else:

            combined = pd.concat(
                [
                    self.history,
                    chunk
                ],
                ignore_index=True
            )

        # ----------------------------------------------------
        # Run normal Candidate C inference
        # ----------------------------------------------------

        result = self.engine.predict(
            combined
        )

        # ----------------------------------------------------
        # Only return results belonging to the NEW chunk
        # ----------------------------------------------------

        history_length = (
            0
            if self.history is None
            else len(self.history)
        )

        new_result = result.iloc[
            history_length:
        ].copy()

        new_result = new_result.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Update persistent history
        #
        # Keep the latest 4 raw SCADA observations.
        # ----------------------------------------------------

        self.history = combined.tail(
            self.HISTORY_SIZE
        ).copy()

        self.history = self.history.reset_index(
            drop=True
        )

        return new_result

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.history = None