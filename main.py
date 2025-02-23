from src.logger import get_logger
from src import training
from src.config import Config
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def main():
    train_cfg = Config("train.toml")
    now = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"{now}_{train_cfg.model.name.split('.')[-1]}"
    logger = get_logger()
    logger.info(f"Starting experiment for {filename}")
    logger.info("Preparing...")
    data = training.prepare_data()
    data.feature_count
    data.row_count
    data.split_data(train_cfg.split.train_size, stratify=data.y)
    data.preprocess_data(StandardScaler())
    data.target_count
    model = training.Model(training.load_model(), data)
    logger.info("Starting training...")
    model.train()
    model.evaluate(filename)
    model.display_confusion_matrix(filename)
    model.feature_importance(filename)
    model.save(filename)
    logger.info("Done training!")


if __name__ == "__main__":
    main()
