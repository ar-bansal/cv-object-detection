import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pytorch_lightning import Trainer
from mlops.ml_logging.loggers import PyTorchLogger


__all__ = ["run_pipeline"]


def create_trainer(n_epochs: int=None) -> Trainer:
    trainer = Trainer(
            max_epochs=n_epochs, 
            logger=False, 
            enable_checkpointing=False
        )
    return trainer


def train(model, datamodule, n_epochs):
    trainer = create_trainer(n_epochs)
    trainer.fit(
        model=model, 
        datamodule=datamodule
        )

    return trainer.model


def evaluate(model, datamodule):
    trainer = create_trainer()

    pred_labels = trainer.predict(
        model=model, 
        datamodule=datamodule
    )

    true_labels = datamodule.test_dataset.targets

    return np.concatenate(pred_labels).reshape(-1, 1), true_labels


def get_metrics(predictions, targets, class_labels=None):
    predicted_classes = np.array(class_labels)[predictions]
    target_classes = np.array(class_labels)[targets]

    cls_report = classification_report(
        target_classes, 
        predicted_classes, 
        target_names=class_labels, 
        output_dict=True
    )

    report_df = pd.DataFrame(cls_report).transpose().round(4)

    true_cols = [f"true_{label}" for label in class_labels]
    pred_idxs = [f"pred_{label}" for label in class_labels]

    cm = confusion_matrix(
        target_classes, 
        predicted_classes, 
        labels=class_labels
    ).T

    cm_df = pd.DataFrame(cm, columns=true_cols, index=pred_idxs)

    return {
        "test_accuracy": round(accuracy_score(target_classes, predicted_classes), 4), 
        "test_classification_report": report_df, 
        "test_confusion_matrix": cm_df
    }


@PyTorchLogger(save_graph=True).log
def train_and_evaluate(model, datamodule, n_epochs, class_labels, experiment_name, input_shape):
    model = train(model, datamodule, n_epochs)

    preds, targets = evaluate(model, datamodule)

    metrics = get_metrics(preds, targets, class_labels=class_labels)

    return model, metrics


def run_pipeline(model, datamodule, num_epochs, experiment_name):
    INPUT_SHAPE = (1, 3, 32, 32)

    class_labels = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
   
    train_and_evaluate(
        model, 
        datamodule, 
        num_epochs, 
        class_labels, 
        experiment_name=experiment_name, 
        input_shape=INPUT_SHAPE
    )