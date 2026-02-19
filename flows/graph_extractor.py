from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import yaml
from blocks.quant_layers import BinaryDense, TernaryDense
from blocks.pruner import ModelPruner


LOGGER = get_run_logger()

@task(name="Fetch Config")
def fetch_config(config_file: str) -> dict:
    LOGGER.info(f"Fetching config from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


@task(name="Data Split")
def get_data_splits():
    LOGGER.info(r"Creating 10 and 50% data splits")
    pass


@task(name="Training Full Precision Model")
def train_full_precision_model(config: dict, X_train, y_train, X_val, y_val, data_percent):
    LOGGER.info(f"Training full precision model on {data_percent}% of data")

    model_config = config["models"]["full"]
    model = Sequential()

    for layer in model_config["layers"]:
        if layer["type"] == "Dense":
            model.add(Dense(
                units=layer["units"],
                activation=layer["activation"],
                input_shape=layer.get("input_shape", None)
            ))
        elif layer["type"] == "Dropout":
            model.add(Dropout(rate=layer["rate"]))
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())

    model.compile(
        optimizer=config["training"]["optimizer"],
        loss=config["training"]["loss"],
        metrics=[Recall(name='recall'), Precision(name='precision')]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        class_weight=config["training"]["class_weights"],
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    return model


@task(name="Training Binary Model")
def train_binary_model(config: dict, X_train, y_train, X_val, y_val):
    LOGGER.info("Training binary model")

    model_config = config["models"]["binary"]
    model = Sequential()

    for layer in model_config["layers"]:
        if layer["type"] == "BinaryDense":
            model.add(BinaryDense(
                units=layer["units"],
                activation=layer["activation"],
                input_shape=layer.get("input_shape", None)
            ))
        elif layer["type"] == "Dropout":
            model.add(Dropout(rate=layer["rate"]))
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())

    model.compile(
        optimizer=config["training"]["optimizer"],
        loss=config["training"]["loss"],
        metrics=[Recall(name='recall'), Precision(name='precision')]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        class_weight=config["training"]["class_weights"],
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    return model


@task(name="Training Ternary Model")
def train_ternary_model(config: dict, X_train, y_train, X_val, y_val):
    LOGGER.info("Training ternary model")

    model_config = config["models"]["ternary"]
    model = Sequential()

    for layer in model_config["layers"]:
        if layer["type"] == "TernaryDense":
            model.add(TernaryDense(
                units=layer["units"],
                activation=layer["activation"],
                input_shape=layer.get("input_shape", None)
            ))
        elif layer["type"] == "Dropout":
            model.add(Dropout(rate=layer["rate"]))
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())

    model.compile(
        optimizer=config["training"]["optimizer"],
        loss=config["training"]["loss"],
        metrics=[Recall(name='recall'), Precision(name='precision')]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        class_weight=config["training"]["class_weights"],
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    return model


@task(name="Prune Model")
def prune_model(model, config, X_train, y_train, X_val, y_val):
    LOGGER.info("Pruning model using tfmot's PolynomialDecay()")

    pruner = ModelPruner()
    pruner.prune_model(model, config)

    pruned_model = pruner.finetune_model(X_train, y_train, X_val, y_val)

    return pruner


@flow(name="Graph Extractor")
def graph_extractor():
    pass