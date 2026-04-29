import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


DATASET_PATH = os.getenv(
    "MODEL_DATASET_PATH",
    "/mnt/user-data/uploads/room_air_quality.csv"
)


class BatchLinearRegression:
    """
    Simple linear regression trained with full-batch gradient descent.
    """

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 2000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.loss_history = []

    def fit(self, X_b: np.ndarray, y: np.ndarray):
        m, n = X_b.shape
        self.theta = np.zeros(n)

        for i in range(self.n_iterations):
            y_hat = X_b @ self.theta
            error = y_hat - y
            gradient = (2 / m) * (X_b.T @ error)

            self.theta -= self.lr * gradient

            if i % 50 == 0:
                loss = np.mean(error ** 2)
                self.loss_history.append(loss)

        return self

    def predict(self, X_b: np.ndarray) -> np.ndarray:
        return X_b @ self.theta


class RoomSatisfactionModel:
    """
    Trains a satisfaction model and uses it to recommend room settings.
    """

    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.model = BatchLinearRegression()
        self.is_trained = False
        self.metrics = {}

        self.feature_cols = [
            "Temperature",
            "Humidity",
            "Light",
            "temp_distance",
            "humidity_distance",
            "light_distance",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "quarter",
        ]

    def train(self):
        """
        Loads data, prepares features, and trains the model.
        """
        df = self._load_data()
        df = self._prepare_data(df)
        df = self._create_satisfaction_target(df)
        df = self._add_model_features(df)

        X = df[self.feature_cols].values
        y = df["satisfaction"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        X_train_b = self._add_bias(X_train_sc)
        X_test_b = self._add_bias(X_test_sc)

        self.model.fit(X_train_b, y_train)

        y_pred_train = self.model.predict(X_train_b)
        y_pred_test = self.model.predict(X_test_b)

        self.metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test),
        }

        self.is_trained = True

    def predict_score(
        self,
        temperature: float,
        humidity: float,
        light: float,
        hour: int,
        day_of_week: int,
        month: int,
        is_weekend: int,
        quarter: int
    ) -> float:
        """
        Predicts satisfaction score for one room scenario.
        """
        if not self.is_trained:
            self.train()

        features = self._build_feature_row(
            temperature=temperature,
            humidity=humidity,
            light=light,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            is_weekend=is_weekend,
            quarter=quarter,
        )

        features_sc = self.scaler.transform(features)
        features_b = self._add_bias(features_sc)

        score = self.model.predict(features_b)[0]

        return float(np.clip(score, 0, 100))

    def recommend_scenario(self, measurements: list[dict]) -> dict:
        """
        Finds the best room settings based on current sensor data.
        """
        if not self.is_trained:
            self.train()

        current = self._get_current_measurement(measurements)
        now = datetime.utcnow()

        hour = now.hour
        day_of_week = now.weekday()
        month = now.month
        is_weekend = int(day_of_week in [5, 6])
        quarter = (month - 1) // 3 + 1

        candidates = self._build_candidate_scenarios(
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            is_weekend=is_weekend,
            quarter=quarter,
        )

        candidates_sc = self.scaler.transform(candidates)
        candidates_b = self._add_bias(candidates_sc)

        scores = self.model.predict(candidates_b)
        best_index = int(np.argmax(scores))

        best = candidates[best_index]

        return {
            "prefTemperature": round(float(best[0]), 1),
            "prefHumidity": round(float(best[1]), 1),
            "prefLight": round(float(best[2]), 1),
        }

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the CSV file.
        """
        return pd.read_csv(self.dataset_path)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset and creates time features.
        """
        df = df[[
            "Timestamp",
            "Temperature (?C)",
            "Humidity (%)",
            "Light Intensity (lux)"
        ]].copy()

        df.columns = [
            "Timestamp",
            "Temperature",
            "Humidity",
            "Light"
        ]

        df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())
        df["Humidity"] = df["Humidity"].fillna(df["Humidity"].mean())
        df["Light"] = df["Light"].fillna(df["Light"].mean())

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)

        df["month"] = df["Timestamp"].dt.month
        df["day_of_week"] = df["Timestamp"].dt.dayofweek
        df["hour"] = df["Timestamp"].dt.hour
        df["quarter"] = df["Timestamp"].dt.quarter
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        df.drop(columns=["Timestamp"], inplace=True)

        for col in ["Temperature", "Humidity", "Light"]:
            df = self._remove_outliers_iqr(df, col)

        return df

    def _create_satisfaction_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the target value used for training.
        """
        temp_score = self._gaussian_comfort(
            df["Temperature"],
            optimal=22.0,
            sigma=2.5
        )

        humidity_score = self._gaussian_comfort(
            df["Humidity"],
            optimal=50.0,
            sigma=10.0
        )

        light_score = self._gaussian_comfort(
            df["Light"],
            optimal=500.0,
            sigma=200.0
        )

        df["satisfaction"] = (
            0.40 * temp_score +
            0.35 * humidity_score +
            0.25 * light_score
        ) * 100

        return df

    def _add_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features that help the model understand comfort distance.
        """
        df["temp_distance"] = abs(df["Temperature"] - 22.0)
        df["humidity_distance"] = abs(df["Humidity"] - 50.0)
        df["light_distance"] = abs(df["Light"] - 500.0)

        return df

    def _build_candidate_scenarios(
        self,
        hour: int,
        day_of_week: int,
        month: int,
        is_weekend: int,
        quarter: int
    ) -> np.ndarray:
        """
        Builds possible room scenarios and lets the model choose the best one.
        """
        temperatures = np.arange(18.0, 26.5, 0.5)
        humidities = np.arange(35.0, 65.5, 2.5)
        lights = np.arange(200.0, 800.0, 50.0)

        rows = []

        for temperature in temperatures:
            for humidity in humidities:
                for light in lights:
                    row = self._build_feature_row(
                        temperature=temperature,
                        humidity=humidity,
                        light=light,
                        hour=hour,
                        day_of_week=day_of_week,
                        month=month,
                        is_weekend=is_weekend,
                        quarter=quarter,
                    )

                    rows.append(row[0])

        return np.array(rows)

    def _build_feature_row(
        self,
        temperature: float,
        humidity: float,
        light: float,
        hour: int,
        day_of_week: int,
        month: int,
        is_weekend: int,
        quarter: int
    ) -> np.ndarray:
        """
        Creates one feature row in the same format as training data.
        """
        return np.array([[
            temperature,
            humidity,
            light,
            abs(temperature - 22.0),
            abs(humidity - 50.0),
            abs(light - 500.0),
            hour,
            day_of_week,
            month,
            is_weekend,
            quarter,
        ]])

    def _get_current_measurement(self, measurements: list[dict]) -> dict:
        """
        Gets the latest sensor measurement.
        """
        if not measurements:
            raise ValueError("No measurements received from IoT service")

        if isinstance(measurements, dict):
            return measurements

        return measurements[-1]

    def _remove_outliers_iqr(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Removes extreme values from one column.
        """
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return data[
            (data[col] >= lower_bound) &
            (data[col] <= upper_bound)
        ]

    def _gaussian_comfort(self, value, optimal: float, sigma: float):
        """
        Gives a high score when the value is close to the optimal value.
        """
        return np.exp(-0.5 * ((value - optimal) / sigma) ** 2)


model = RoomSatisfactionModel()


def predict_scenario(measurements: list[dict]) -> dict:
    """
    Main function used by the API.
    """
    return model.recommend_scenario(measurements)