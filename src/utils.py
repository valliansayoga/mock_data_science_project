import sqlalchemy as sa
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import get_logger

logger = get_logger()


class Data:
    def __init__(self, tablename: str, engine: sa.engine.base.Engine, target: str):
        logger.info(f"Reading data from {tablename}")
        data = read_db(tablename, engine)
        self.target = target
        self.X = data.drop(target, axis=1)
        self.y = data[target]
        logger.info("Done")

    def split_data(
        self,
        train_size: float,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """
        Split data into train and test sets.
        return: X_train, X_test, y_train, y_test
        """
        logger.info(
            f"Splitting data into train ({train_size:.2f}) and test sets ({1 - train_size:.2f})"
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=train_size, random_state=42, **kwargs
        )
        logger.info("Done")
        return

    def preprocess_data(self, preprocessor):
        """
        Scale the features of the data.
        """
        logger.info("Scaling the features of the data")
        self.scaler = preprocessor
        self.X_train = preprocessor.fit_transform(self.X_train)
        self.X_test = preprocessor.transform(self.X_test)
        logger.info("Done")
        return

    @property
    def target_count(self):
        """
        Count the target classes.
        """
        ys = ("y", "y_train", "y_test")
        for y in ys:
            obj = getattr(self, y, None)
            if obj is None:
                continue
            counts = obj.value_counts().to_dict()
            logger.info(f"Count of {y}: {counts}")
        return

    @property
    def feature_count(self):
        """
        Count the features.
        """
        logger.info(f"Count of features: {self.X.shape[1]}")
        return

    @property
    def row_count(self):
        """
        Rows the target classes.
        """
        logger.info(f"Count of rows: {self.X.shape[0]}")
        return


def create_engine(uri: str) -> sa.engine.base.Engine:
    """
    Create a SQLAlchemy engine.
    return: sqlalchemy.engine.base.Engine
    """
    return sa.create_engine(uri)


def write_db(df: pd.DataFrame, table_name: str, engine: sa.engine.base.Engine):
    """
    Write a DataFrame to a SQL database using SQLAlchemy.
    """
    with engine.connect() as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    return


def read_db(table_name: str, engine: sa.engine.base.Engine, **kwargs) -> pd.DataFrame:
    """
    Read a SQL table into a DataFrame using SQLAlchemy.
    return pandas.DataFrame
    """
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn, **kwargs)
    return df


def check_basic_info(df: pd.DataFrame):
    """
    Check the basic information of the DataFrame.
    """
    shape = [df.shape]
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    form = pd.DataFrame()
    form["shape"] = shape
    form["nulls"] = nulls
    print(
        "DF shape & nulls\n", "----------------\n", form.T, "\n", "----------------\n"
    )
    print()

    print("DF Dtypes\n", "---------n", df.dtypes, "\n", "---------n")

    description = df.describe()
    print(
        "DF description\n",
        "----------------\n",
        description,
        "\n",
        "----------------\n",
    )
    return
