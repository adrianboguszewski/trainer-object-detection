import types

import mlflow


def safe_index(arr, idx):
    return arr[idx] if 0 <= idx < len(arr) else None


class MetricsTensorBoardSinkMLflow:
    """
    Replacement for MetricsTensorBoardSink that logs to MLflow instead of TensorBoard.
    Keeps the same interface: __init__, update, close.
    """

    def __init__(self, output_dir: str):
        print("MLflow Metrics sink initialized")

    def update(self, values: dict):
        epoch = values["epoch"]

        # losses
        if "train_loss" in values:
            mlflow.log_metric("Loss/Train", values["train_loss"], step=epoch)
        if "test_loss" in values:
            mlflow.log_metric("Loss/Test", values["test_loss"], step=epoch)

        # standard COCO eval
        if "test_coco_eval_bbox" in values:
            coco_eval = values["test_coco_eval_bbox"]
            ap50_90 = safe_index(coco_eval, 0)
            ap50 = safe_index(coco_eval, 1)
            ar50_90 = safe_index(coco_eval, 8)
            if ap50_90 is not None:
                mlflow.log_metric("Metrics/Base/AP50_90", ap50_90, step=epoch)
            if ap50 is not None:
                mlflow.log_metric("Metrics/Base/AP50", ap50, step=epoch)
            if ar50_90 is not None:
                mlflow.log_metric("Metrics/Base/AR50_90", ar50_90, step=epoch)

        # EMA COCO eval
        if "ema_test_coco_eval_bbox" in values:
            ema_coco_eval = values["ema_test_coco_eval_bbox"]
            ema_ap50_90 = safe_index(ema_coco_eval, 0)
            ema_ap50 = safe_index(ema_coco_eval, 1)
            ema_ar50_90 = safe_index(ema_coco_eval, 8)
            if ema_ap50_90 is not None:
                mlflow.log_metric("Metrics/EMA/AP50_90", ema_ap50_90, step=epoch)
            if ema_ap50 is not None:
                mlflow.log_metric("Metrics/EMA/AP50", ema_ap50, step=epoch)
            if ema_ar50_90 is not None:
                mlflow.log_metric("Metrics/EMA/AR50_90", ema_ar50_90, step=epoch)

    def save(self):
        pass

    def close(self):
        pass


def patch_to_support_experiment_tracker_with_hafnia(detr: types.ModuleType):
    detr.MetricsPlotSink = MetricsTensorBoardSinkMLflow

    return detr
