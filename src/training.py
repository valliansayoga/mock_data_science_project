import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from src import utils
from src.config import Config, get_config
from src.logger import get_logger


logger = get_logger()
cfg = get_config()
root_dir = Path(cfg.paths.pythonpath)
train_cfg = Config(root_dir / "train.toml")
engine = utils.create_engine(cfg.general.db_uri)


class Model:
    def __init__(self, model=None, data=None):
        self.model = model
        self.data = data

        if not hasattr(self.data, "y_train"):
            self.data.split_data(train_cfg.split.train_size)
        if not hasattr(self.data, "preprocessor"):
            logger.info("Data is not preprocessed. Please preprocess the data.")

    def train(self):
        self.model.fit(self.data.X_train, self.data.y_train)
        return self.model

    def evaluate(self, filename: str):
        save_path = Path(train_cfg.paths.metrics) / filename
        # save_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.metrics()
        cr = self.classification_report()
        cm = self.confusion_matrix()
        metrics.to_csv(
            _process_path(save_path, "_metrics", ".csv"),
            index=True,
        )
        cm.to_csv(_process_path(save_path, "_cm", ".csv"), index=True)

        with open(_process_path(save_path, "_cr", ".txt"), "w") as f:
            f.write(cr)

        logger.info(f"Evaluation saved to {save_path.as_posix()}")
        return

    def classification_report(self) -> str:
        y_pred = self.predict(self.data.X_test)
        cr = classification_report(self.data.y_test, y_pred)
        return cr

    def predict(self, X) -> np.array:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.array:
        if not hasattr(self.model, "predict_proba"):
            logger.error(f"Model {self.model} does not have predict_proba method")
            raise AttributeError(
                f"Model {self.model} does not have predict_proba method"
            )
        return self.model.predict_proba(X)

    def confusion_matrix(self) -> pd.DataFrame:
        y_pred = self.predict(self.data.X_test)
        cm = confusion_matrix(self.data.y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, columns=self.data.y_test.unique(), index=self.data.y_test.unique()
        )
        return cm_df

    def display_confusion_matrix(self, filename):
        save_path = Path(train_cfg.paths.images) / filename
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        cm = self.confusion_matrix()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm.to_numpy())
        disp.plot().figure_.savefig(_process_path(save_path, "_cm", ".png"))

    def metrics(self) -> pd.DataFrame:
        y_pred = self.predict(self.data.X_test)
        metrics = {
            "accuracy": accuracy_score(self.data.y_test, y_pred),
            "f1": f1_score(self.data.y_test, y_pred),
            "roc_auc": roc_auc_score(self.data.y_test, y_pred),
            "precision": precision_score(self.data.y_test, y_pred),
            "recall": recall_score(self.data.y_test, y_pred),
        }
        return pd.DataFrame(metrics, index=["metrics"]).T

    def feature_importance(self, filename):
        try:
            save_path = Path(train_cfg.paths.metrics) / filename
            importances = self.model.feature_importances_
            features = self.data.X.columns
            importance_df = pd.DataFrame(
                importances, index=features, columns=["importance"]
            )
            importance_df.sort_values(by="importance", ascending=False, inplace=True)
            importance_df.to_csv(
                _process_path(save_path, "_importance", ".csv"),
                index=True,
            )
        except AttributeError:
            logger.error(f"Model {self.model} does not have feature importances")
            raise AttributeError(
                f"Model {self.model} does not have feature importances"
            )
        return importances

    def save(self, filename):
        save_path = Path(train_cfg.paths.models) / filename
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_process_path(save_path, "", ".pkl"), "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {save_path.as_posix()}")
        return

    def load(self, filename):
        with open(filename, "rb") as f:
            self.model = pickle.load(f)

        return self.model


def prepare_data() -> utils.Data:
    data = utils.Data(cfg.tables.churn, engine, "target")
    data.split_data(train_cfg.split.train_size)
    data.preprocess_data(StandardScaler())
    data.target_count
    return data


def load_model():
    try:
        name = train_cfg.model.name
        parts = name.split(".")
        model = getattr(sklearn, parts[0], None)
        model = getattr(model, parts[1], None)
        model = model(**train_cfg.params)
    except AttributeError:
        logger.error(f"Model {name} not found in sklearn")
        raise AttributeError(f"Model {name} not found in sklearn")
    except TypeError:
        logger.error(f"Model {name} not found in sklearn")
        raise AttributeError(f"Model {name} not found in sklearn")

    return model


def _process_path(path: Path, typ: str = "", suffix: str = ".png") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path / path.with_name(path.stem + typ).with_suffix(suffix).name
