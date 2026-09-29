import os
import joblib
import numpy as np
import pandas as pd


class CandidateCInferenceEngine:
    """
    Reusable inference engine for HAI 23.05 Candidate C.

    Candidate C schema is loaded from the authoritative
    Stage 24B feature manifest.

    Candidate C:
        58 original SCADA features
        30 abs_diff_1s features
        30 rolling_std_5s features
        --------------------------------
        118 total features
    """

    def __init__(self, model_path, manifest_path):

        self.model_path = model_path
        self.manifest_path = manifest_path

        # ====================================================
        # Validate files
        # ====================================================

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Feature manifest not found: {manifest_path}"
            )

        # ====================================================
        # Load model
        # ====================================================

        self.model = joblib.load(model_path)

        # ====================================================
        # Load authoritative feature manifest
        # ====================================================

        manifest = pd.read_csv(manifest_path)

        required_columns = [
            "feature_order",
            "feature"
        ]

        for column in required_columns:

            if column not in manifest.columns:
                raise ValueError(
                    f"Required manifest column missing: {column}"
                )

        # ----------------------------------------------------
        # Sort explicitly by authoritative feature order
        # ----------------------------------------------------

        manifest = manifest.sort_values(
            "feature_order"
        ).reset_index(drop=True)

        self.feature_names = (
            manifest["feature"]
            .astype(str)
            .tolist()
        )

        # ====================================================
        # Validate total feature count
        # ====================================================

        if len(self.feature_names) != 118:
            raise ValueError(
                f"Expected 118 Candidate C features, "
                f"found {len(self.feature_names)}"
            )

        if len(set(self.feature_names)) != 118:
            raise ValueError(
                "Candidate C manifest contains duplicate "
                "feature names."
            )

        # ====================================================
        # Separate feature families
        # ====================================================

        self.original_features = [
            feature
            for feature in self.feature_names
            if "__" not in feature
        ]

        self.abs_diff_features = [
            feature
            for feature in self.feature_names
            if "__abs_diff_1s" in feature
        ]

        self.rolling_features = [
            feature
            for feature in self.feature_names
            if "__rolling_std_5s" in feature
        ]

        # ====================================================
        # Validate Candidate C composition
        # ====================================================

        if len(self.original_features) != 58:
            raise ValueError(
                "Candidate C must contain 58 original features. "
                f"Found {len(self.original_features)}."
            )

        if len(self.abs_diff_features) != 30:
            raise ValueError(
                "Candidate C must contain 30 abs_diff_1s "
                "features. "
                f"Found {len(self.abs_diff_features)}."
            )

        if len(self.rolling_features) != 30:
            raise ValueError(
                "Candidate C must contain 30 rolling_std_5s "
                "features. "
                f"Found {len(self.rolling_features)}."
            )

        # ====================================================
        # Validate family total
        # ====================================================

        assert (
            len(self.original_features)
            + len(self.abs_diff_features)
            + len(self.rolling_features)
            == 118
        )

        # ====================================================
        # Validate temporal feature mappings
        # ====================================================

        for feature in self.abs_diff_features:

            base = feature.replace(
                "__abs_diff_1s",
                ""
            )

            if base not in self.original_features:

                raise ValueError(
                    f"Temporal feature {feature} refers to "
                    f"missing original feature {base}"
                )

        for feature in self.rolling_features:

            base = feature.replace(
                "__rolling_std_5s",
                ""
            )

            if base not in self.original_features:

                raise ValueError(
                    f"Temporal feature {feature} refers to "
                    f"missing original feature {base}"
                )

    # ========================================================
    # BUILD CANDIDATE C FEATURES
    # ========================================================

    def build_features(self, data):

        df = data.copy()

        # ----------------------------------------------------
        # Check original SCADA features
        # ----------------------------------------------------

        missing = [
            feature
            for feature in self.original_features
            if feature not in df.columns
        ]

        if missing:

            raise ValueError(
                "Missing original SCADA features:\n"
                + "\n".join(missing)
            )

        # ----------------------------------------------------
        # Convert original features to numeric
        # ----------------------------------------------------

        for feature in self.original_features:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # ====================================================
        # Absolute first differences
        # ====================================================

        for feature in self.abs_diff_features:

            base = feature.replace(
                "__abs_diff_1s",
                ""
            )

            df[feature] = (
                df[base]
                .diff()
                .abs()
            )

        # ====================================================
        # Five-second rolling standard deviation
        # ====================================================

        for feature in self.rolling_features:

            base = feature.replace(
                "__rolling_std_5s",
                ""
            )

            df[feature] = (
                df[base]
                .rolling(
                    window=5,
                    min_periods=5
                )
                .std()
            )

        # ====================================================
        # Authoritative Candidate C ordering
        # ====================================================

        output_columns = [
            column
            for column in ["timestamp"] + self.feature_names
            if column in df.columns
        ]

        return df[output_columns]

    # ========================================================
    # RUN INFERENCE
    # ========================================================

    def predict(self, data):

        features = self.build_features(data)

        X = features[
            self.feature_names
        ]

        # ----------------------------------------------------
        # Valid rows
        # ----------------------------------------------------

        valid_mask = X.notna().all(axis=1)

        output = pd.DataFrame(
            index=features.index
        )

        # ----------------------------------------------------
        # Preserve timestamp
        # ----------------------------------------------------

        if "timestamp" in features.columns:

            output["timestamp"] = (
                features["timestamp"]
            )

        # ----------------------------------------------------
        # Default state
        # ----------------------------------------------------

        output["anomaly_score"] = np.nan

        output["prediction"] = np.nan

        output["status"] = (
            "INSUFFICIENT_HISTORY"
        )

        # ====================================================
        # Inference
        # ====================================================

        if valid_mask.any():

            X_valid = X.loc[
                valid_mask,
                self.feature_names
            ]

            scores = self.model.decision_function(
                X_valid.to_numpy()
            )

            predictions = (
                scores < 0
            ).astype(int)

            output.loc[
                valid_mask,
                "anomaly_score"
            ] = scores

            output.loc[
                valid_mask,
                "prediction"
            ] = predictions

            output.loc[
                valid_mask,
                "status"
            ] = np.where(
                predictions == 1,
                "ANOMALY",
                "NORMAL"
            )

        return output.reset_index(drop=True)