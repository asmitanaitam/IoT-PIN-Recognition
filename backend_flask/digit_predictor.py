import numpy as np
import joblib

class DigitPredictor:
    def __init__(self):
        # Load models
        self.rf = joblib.load("model_rf.pkl")
        self.knn = joblib.load("model_knn.pkl")
        self.xgb = joblib.load("model_xgb.pkl")

        # Load feature definitions
        self.feature_cols = joblib.load("feature_cols.pkl")
        self.context_size = joblib.load("context_size.pkl")

    def _compute_features(self, df):
        """Compute feature engineering exactly as during training."""
        df = df.copy()

        df["acc_mag"] = np.sqrt(df.ax**2 + df.ay**2 + df.az**2)
        df["gyro_mag"] = np.sqrt(df.gx**2 + df.gy**2 + df.gz**2)

        df["ax2"] = df.ax**2
        df["ay2"] = df.ay**2
        df["az2"] = df.az**2

        df["ax_ay"] = df.ax * df.ay
        df["ay_az"] = df.ay * df.az
        df["gx_gy"] = df.gx * df.gy
        df["gy_gz"] = df.gy * df.gz

        df["ax_diff"] = df.ax.diff().fillna(0)
        df["ay_diff"] = df.ay.diff().fillna(0)
        df["az_diff"] = df.az.diff().fillna(0)
        df["gx_diff"] = df.gx.diff().fillna(0)
        df["gy_diff"] = df.gy.diff().fillna(0)
        df["gz_diff"] = df.gz.diff().fillna(0)

        df["ax_roll"] = df.ax.rolling(3, center=True).mean()
        df["ay_roll"] = df.ay.rolling(3, center=True).mean()
        df["gz_roll"] = df.gz.rolling(3, center=True).std()

        df = df.fillna(method="bfill").fillna(method="ffill")
        return df[self.feature_cols]

    def _make_context_window(self, features):
        """Flatten last N rows into one context window."""
        window = features.tail(self.context_size).values.flatten()
        return window.reshape(1, -1)

    def predict_digit(self, df):
        """Main prediction API."""
        feats = self._compute_features(df)
        window = self._make_context_window(feats)

        pred_rf = self.rf.predict(window)[0]
        pred_knn = self.knn.predict(window)[0]
        pred_xgb = self.xgb.predict(window)[0]

        votes = np.array([pred_rf, pred_knn, pred_xgb])
        (final_pred, count) = np.unique(votes, return_counts=True)

        return final_pred[np.argmax(count)]
