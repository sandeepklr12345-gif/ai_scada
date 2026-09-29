import os
import argparse
import joblib
import numpy as np
import pandas as pd


class HAICandidateCInference:
    """
    Production inference engine for HAI 23.05 Candidate C.

    Candidate C:
        58 original SCADA features
        30 abs_diff_1s features
        30 rolling_std_5s features
        --------------------------------
        118 total features

    Supports:
        1. Batch inference
        2. Stateful streaming inference
    """

    HISTORY_SIZE = 4

    def __init__(
        self,
        model_path,
        manifest_path
    ):

        # ====================================================
        # LOAD MODEL
        # ====================================================

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        self.model = joblib.load(
            model_path
        )

        # ====================================================
        # LOAD AUTHORITATIVE MANIFEST
        # ====================================================

        manifest = pd.read_csv(
            manifest_path
        )

        required_columns = [
            "feature_order",
            "feature"
        ]

        for column in required_columns:

            if column not in manifest.columns:

                raise ValueError(
                    f"Missing manifest column: {column}"
                )

        manifest = manifest.sort_values(
            "feature_order"
        ).reset_index(drop=True)

        self.feature_names = (
            manifest["feature"]
            .astype(str)
            .tolist()
        )

        # ====================================================
        # VALIDATE SCHEMA
        # ====================================================

        if len(self.feature_names) != 118:

            raise ValueError(
                f"Expected 118 features, "
                f"found {len(self.feature_names)}"
            )

        if len(set(self.feature_names)) != 118:

            raise ValueError(
                "Duplicate feature names found."
            )

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

        if len(self.original_features) != 58:
            raise ValueError(
                "Expected 58 original features."
            )

        if len(self.abs_diff_features) != 30:
            raise ValueError(
                "Expected 30 abs_diff_1s features."
            )

        if len(self.rolling_features) != 30:
            raise ValueError(
                "Expected 30 rolling_std_5s features."
            )

        # ====================================================
        # RESET STREAMING STATE
        # ====================================================

        self.reset()

    # ========================================================
    # RESET STREAMING HISTORY
    # ========================================================

    def reset(self):

        self.history = pd.DataFrame(
            columns=self.original_features
        )

        self.previous_timestamp = None

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    def _validate_input(
        self,
        data
    ):

        missing = [
            feature
            for feature in self.original_features
            if feature not in data.columns
        ]

        if missing:

            raise ValueError(
                "Missing SCADA features:\n"
                + "\n".join(missing)
            )

    # ========================================================
    # BUILD FEATURES
    # ========================================================

    @staticmethod
    def _rolling_std_5(series):
        """
        Deterministic 5-sample rolling sample standard deviation.

        Uses the exact same five raw values for every window instead of
        pandas' rolling implementation. This makes batch and stateful
        chunked inference numerically identical at chunk boundaries.
        """
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        result = np.full(values.shape[0], np.nan, dtype=np.float64)

        if len(values) < 5:
            return pd.Series(result, index=series.index)

        windows = np.lib.stride_tricks.sliding_window_view(values, 5)
        valid = np.isfinite(windows).all(axis=1)

        if valid.any():
            # Sample standard deviation, matching pandas rolling().std()
            # with its default ddof=1.
            valid_windows = windows[valid]
            means = valid_windows.mean(axis=1)
            centered = valid_windows - means[:, None]
            result_indices = np.flatnonzero(valid) + 4
            result[result_indices] = np.sqrt(
                np.sum(centered * centered, axis=1) / 4.0
            )

        return pd.Series(result, index=series.index)

    def _build_features(
        self,
        data
    ):

        df = data.copy()

        self._validate_input(
            df
        )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        for feature in self.original_features:

            df[feature] = pd.to_numeric(
                df[feature],
                errors="coerce"
            )

        # ----------------------------------------------------
        # First differences
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Rolling 5-second standard deviation
        # ----------------------------------------------------

        for feature in self.rolling_features:

            base = feature.replace(
                "__rolling_std_5s",
                ""
            )

            df[feature] = self._rolling_std_5(
                df[base]
            )

        return df
     
    # ========================================================
    # CANDIDATE C FEATURE MATRIX
    # ========================================================

    def build_candidate_features(self, data):
        """
        Build the complete 118-feature Candidate C matrix.

        Used for internal batch/streaming validation.
        """

        features = self._build_features(data)

        return features[
            self.feature_names
        ].copy()
    # ========================================================
    # BATCH INFERENCE
    # ========================================================

    def predict_batch(
        self,
        data
    ):

        features = self._build_features(
            data
        )

        X = features[
            self.feature_names
        ]

        valid_mask = (
            X.notna().all(axis=1)
        )

        output = pd.DataFrame(
            index=features.index
        )

        if "timestamp" in features.columns:

            output["timestamp"] = (
                features["timestamp"]
            )

        output["anomaly_score"] = np.nan

        output["prediction"] = np.nan

        output["status"] = (
            "INSUFFICIENT_HISTORY"
        )

        if valid_mask.any():

            X_valid = X.loc[
                valid_mask,
                self.feature_names
            ]

            scores = (
                self.model
                .decision_function(
                    X_valid.to_numpy()
                )
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

        return output.reset_index(
            drop=True
        )

    # ========================================================
    # STATEFUL STREAMING INFERENCE
    # ========================================================

    def predict_stream(
        self,
        chunk
    ):

        chunk = chunk.copy()

        self._validate_input(
            chunk
        )

        for feature in self.original_features:
            chunk[feature] = pd.to_numeric(
                chunk[feature], errors="coerce"
            )

        # ----------------------------------------------------
        # Keep timestamp separate
        # ----------------------------------------------------

        timestamp_column = None

        if "timestamp" in chunk.columns:

            timestamp_column = (
                chunk["timestamp"]
                .copy()
            )

        # ----------------------------------------------------
        # Build continuous raw data
        #
        # Previous 4 rows + current chunk
        # ----------------------------------------------------

        if len(self.history) > 0:

            combined = pd.concat(
                [
                    self.history,
                    chunk[
                        self.original_features
                    ]
                ],
                ignore_index=True
            )

        else:

            combined = chunk[
                self.original_features
            ].copy()

        # ----------------------------------------------------
        # Calculate temporal features over the continuous
        # history + current chunk.
        # ----------------------------------------------------

        features = combined.copy()

        # ----------------------------------------------------
        # Absolute differences
        # ----------------------------------------------------

        for feature in self.abs_diff_features:

            base = feature.replace(
                "__abs_diff_1s",
                ""
            )

            features[feature] = (
                features[base]
                .diff()
                .abs()
            )

        # ----------------------------------------------------
        # Rolling standard deviation
        # ----------------------------------------------------

        for feature in self.rolling_features:

            base = feature.replace(
                "__rolling_std_5s",
                ""
            )

            features[feature] = self._rolling_std_5(
                features[base]
            )

        # ----------------------------------------------------
        # Remove history rows.
        # ----------------------------------------------------

        history_length = len(
            self.history
        )

        current_features = features.iloc[
        history_length:
        ].copy().reset_index(drop=True)

        # ----------------------------------------------------
        # Restore timestamp
        # ----------------------------------------------------

        if timestamp_column is not None:

            current_features.insert(
                0,
                "timestamp",
                chunk["timestamp"].reset_index(drop=True)
                )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        X = current_features[
            self.feature_names
        ]

        valid_mask = (
            X.notna().all(axis=1)
        )

        output = pd.DataFrame(
            index=current_features.index
        )

        if "timestamp" in current_features.columns:

            output["timestamp"] = (
                current_features[
                    "timestamp"
                ]
            )

        output["anomaly_score"] = np.nan

        output["prediction"] = np.nan

        output["status"] = (
            "INSUFFICIENT_HISTORY"
        )

        if valid_mask.any():

            X_valid = X.loc[
                valid_mask,
                self.feature_names
            ]

            scores = (
                self.model
                .decision_function(
                    X_valid.to_numpy()
                )
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

        # ----------------------------------------------------
        # Update persistent history
        #
        # Keep only the latest 4 RAW SCADA rows.
        # ----------------------------------------------------

        self.history = (
            combined
            .tail(
                self.HISTORY_SIZE
            )
            .copy()
            .reset_index(drop=True)
        )

        return output.reset_index(
            drop=True
        )


# ============================================================
# SINGLE-FILE CLI
# ============================================================

def run_validation(engine, test_path, chunk_size=1000):
    print("=" * 70)
    print("CANDIDATE C BATCH vs STREAMING VALIDATION")
    print("=" * 70)
    data = pd.read_csv(test_path)
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    print(f"Rows loaded: {len(data):,}")
    print(f"Chunk size: {chunk_size:,}")

    print("\nRunning batch inference...")
    batch = engine.predict_batch(data)
    print("Running stateful streaming inference...")
    engine.reset()
    parts=[]
    for start in range(0,len(data),chunk_size):
        parts.append(engine.predict_stream(data.iloc[start:start+chunk_size].copy()))
    stream=pd.concat(parts,ignore_index=True)

    if len(batch)!=len(stream): raise AssertionError("Row count mismatch")
    batch_ts = pd.to_datetime(
    batch["timestamp"],
    errors="coerce"
    ).reset_index(drop=True)

    stream_ts = pd.to_datetime(
        stream["timestamp"],
        errors="coerce"
    ).reset_index(drop=True)

    if not batch_ts.equals(stream_ts):
        timestamp_diff = batch_ts != stream_ts

        print("\nTimestamp mismatch detected")
        print(
            "Different timestamps:",
            int(timestamp_diff.sum())
        )

        first_diff = np.flatnonzero(
            timestamp_diff.to_numpy()
        )

        if len(first_diff) > 0:
            i = int(first_diff[0])
            print("First mismatch row:", i)
            print("Batch :", batch_ts.iloc[i])
            print("Stream:", stream_ts.iloc[i])

        raise AssertionError(
            "Timestamp values genuinely differ."
        )
    bs=pd.to_numeric(batch["anomaly_score"],errors="coerce").to_numpy()
    ss=pd.to_numeric(stream["anomaly_score"],errors="coerce").to_numpy()
    valid=np.isfinite(bs)&np.isfinite(ss)
    diff=np.abs(bs-ss)
    max_diff=float(np.max(diff[valid])) if valid.any() else np.nan
    mean_diff=float(np.mean(diff[valid])) if valid.any() else np.nan
    differing=int(np.sum(valid&(diff>1e-12)))
    pred_mismatch=int(np.sum(batch["prediction"].fillna(-1).to_numpy()!=stream["prediction"].fillna(-1).to_numpy()))
    status_mismatch=int(np.sum(batch["status"].to_numpy()!=stream["status"].to_numpy()))
    bv=int(batch["anomaly_score"].notna().sum()); sv=int(stream["anomaly_score"].notna().sum())
    print("\n"+"-"*70); print("VALIDATION RESULTS"); print("-"*70)
    print(f"Batch rows:                  {len(batch):,}")
    print(f"Streaming rows:              {len(stream):,}")
    print(f"Batch valid rows:            {bv:,}")
    print(f"Streaming valid rows:        {sv:,}")
    print(f"Maximum score difference:    {max_diff:.15f}")
    print(f"Mean score difference:       {mean_diff:.15f}")
    print(f"Rows with score difference:  {differing:,}")
    print(f"Prediction mismatches:       {pred_mismatch:,}")
    print(f"Status mismatches:           {status_mismatch:,}")
    if differing:
        i=int(np.flatnonzero(valid&(diff>1e-12))[0]); print("\nFirst score mismatch:"); print(f"Row:             {i}"); print(f"Timestamp:       {batch.loc[i,'timestamp']}"); print(f"Batch score:     {bs[i]:.15f}"); print(f"Stream score:    {ss[i]:.15f}"); print(f"Absolute diff:   {diff[i]:.15f}")
    passed=(len(batch)==len(stream) and batch["timestamp"].equals(stream["timestamp"]) and bv==sv and max_diff<=1e-12 and pred_mismatch==0 and status_mismatch==0)
    print("\n"+"-"*70); print("VALIDATION: PASS" if passed else "VALIDATION: FAIL"); print("="*70)
    return passed

def run_diagnose(engine, test_path, chunk_size=1000):

    print("=" * 70)
    print("CANDIDATE C FEATURE DIAGNOSTIC")
    print("=" * 70)

    data = pd.read_csv(test_path)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

    print(f"Rows loaded: {len(data):,}")
    print(f"Chunk size: {chunk_size:,}")

    # Batch feature matrix
    batch_features = engine.build_candidate_features(data)

    # Streaming feature matrix
    engine.reset()

    stream_parts = []

    for start in range(0, len(data), chunk_size):

        chunk = data.iloc[
            start:start + chunk_size
        ].copy()

        if len(engine.history) > 0:
            combined = pd.concat(
                [
                    engine.history,
                    chunk[engine.original_features]
                ],
                ignore_index=True
            )
        else:
            combined = chunk[
                engine.original_features
            ].copy()

        features = combined.copy()

        for feature in engine.abs_diff_features:

            base = feature.replace(
                "__abs_diff_1s",
                ""
            )

            features[feature] = (
                features[base]
                .diff()
                .abs()
            )

        for feature in engine.rolling_features:

            base = feature.replace(
                "__rolling_std_5s",
                ""
            )

            features[feature] = self._rolling_std_5(
                features[base]
            )

        history_length = len(
            engine.history
        )

        current = features.iloc[
            history_length:
        ].copy().reset_index(drop=True)

        stream_parts.append(
            current[
                engine.feature_names
            ].copy()
        )

        engine.history = (
            combined
            .tail(engine.HISTORY_SIZE)
            .copy()
            .reset_index(drop=True)
        )

    stream_features = pd.concat(
        stream_parts,
        ignore_index=True
    )

    print("\nComparing feature matrices...")

    if batch_features.shape != stream_features.shape:
        raise AssertionError(
            f"Shape mismatch: "
            f"batch={batch_features.shape}, "
            f"stream={stream_features.shape}"
        )

    differences = (
        batch_features
        - stream_features
    ).abs()

    differences = differences.replace(
        [np.inf, -np.inf],
        np.nan
    )

    max_difference = (
        differences.max()
        .sort_values(
            ascending=False
        )
    )

    print("\nTop 20 feature differences:")
    print("-" * 70)

    for feature, value in max_difference.head(20).items():

        count = int(
            (differences[feature] > 1e-12)
            .sum()
        )

        print(
            f"{feature:<45} "
            f"max={value:.15f} "
            f"count={count:,}"
        )

    total_cells = (
        differences.shape[0]
        * differences.shape[1]
    )

    differing_cells = int(
        (differences > 1e-12)
        .sum()
        .sum()
    )

    print("\n" + "-" * 70)
    print(
        f"Total feature cells:       {total_cells:,}"
    )
    print(
        f"Differing feature cells:   {differing_cells:,}"
    )
    print(
        f"Maximum feature difference: "
        f"{differences.max().max():.15f}"
    )

    # First mismatch
    mismatch = differences > 1e-12

    if mismatch.any().any():

        locations = np.argwhere(
            mismatch.to_numpy()
        )

        row, column = locations[0]

        feature = (
            differences.columns[column]
        )

        print("\nFirst feature mismatch:")
        print(
            f"Row:        {row}"
        )
        print(
            f"Timestamp:  {data.iloc[row]['timestamp']}"
        )
        print(
            f"Feature:    {feature}"
        )
        print(
            f"Batch:      "
            f"{batch_features.iloc[row, column]:.15f}"
        )
        print(
            f"Stream:     "
            f"{stream_features.iloc[row, column]:.15f}"
        )
        print(
            f"Difference: "
            f"{differences.iloc[row, column]:.15f}"
        )

    print("=" * 70)
    return True


if __name__ == "__main__":
    BASE_DIR=r"C:\Users\sandeep\OneDrive\Documents\Ai_Scada"
    FINAL_DIR=os.path.join(BASE_DIR,"data","features","hai","hai-23.05","temporal_representation","final_candidate")
    MODEL_PATH=os.path.join(FINAL_DIR,"hai_2305_candidate_C_isolation_forest.joblib")
    MANIFEST_PATH=os.path.join(FINAL_DIR,"hai_2305_candidate_C_feature_manifest.csv")
    TEST1_PATH=os.path.join(BASE_DIR,"data","features","hai","hai-23.05","temporal_representation","hai-test1_temporal_model_ready.csv")
    parser=argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1000)
    args=parser.parse_args()
    print("="*70); print("HAI CANDIDATE C INFERENCE ENGINE"); print("="*70)
    engine=HAICandidateCInference(MODEL_PATH,MANIFEST_PATH)
    print("Model loaded"); print(f"Feature count: {len(engine.feature_names)}"); print(f"Original features: {len(engine.original_features)}"); print(f"abs_diff_1s features: {len(engine.abs_diff_features)}"); print(f"rolling_std_5s features: {len(engine.rolling_features)}"); print("Engine initialization: PASS")
    if args.diagnose:
        if not os.path.exists(TEST1_PATH):
            raise FileNotFoundError(TEST1_PATH)
        run_diagnose(
            engine,
            TEST1_PATH,
            args.chunk_size
        )

    if args.validate:
        if not os.path.exists(TEST1_PATH):
            raise FileNotFoundError(TEST1_PATH)
        run_validation(
            engine,
            TEST1_PATH,
            args.chunk_size
        )

    if not args.diagnose and not args.validate:
        print("Use --validate or --diagnose.")

    print("="*70)
