from sklearn.datasets import make_classification
import pandas as pd
from src.config import get_config
from src.logger import get_logger
from src.utils import create_engine, write_db


logger = get_logger()
cfg = get_config()
engine = create_engine(cfg.general.db_uri)

logger.info("Creating mock dataset")
X, y = make_classification(n_samples=cfg.general.db_rows, weights=(0.7, 0.3), random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y
logger.info("Done creation")

logger.info("Writing to database")
write_db(df, cfg.tables.churn, engine)
logger.info("Done writing")
