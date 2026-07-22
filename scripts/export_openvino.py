import shutil
from pathlib import Path
from typing import Annotated, Optional

import openvino as ov
import torch
from cyclopts import App, Parameter
from hafnia.experiment import HafniaLogger
from hafnia.experiment.command_builder import auto_save_command_builder_schema
from hafnia.log import user_logger

from trainer_object_detection import utils
from trainer_object_detection.wrapped_model import InferenceConfig, WrappedModel

app = App(name="export_openvino", help="Export RF-DETR model to OpenVINO IR")

""" OpenVINO export examples
# Export the default pretrained model to OpenVINO IR
python scripts/export_openvino.py

# Export a trained checkpoint with a dynamic batch dimension
python scripts/export_openvino.py --model-path ./local_stuff/checkpoint_best_ema.zip --dynamic-batch
"""


@app.default
def main(
    model_path: Annotated[
        str,
        Parameter(
            help=(
                "Path to the model archive (.zip) to export. Note: this is ignored when a checkpoint "
                "is available (e.g. a checkpoint selected for the experiment on the Hafnia platform) - "
                "the checkpoint is exported instead of this model."
            )
        ),
    ] = "./pretrained_models/RFDETRNano.zip",
    opset_version: Annotated[int, Parameter(help="ONNX opset version to target for the intermediate ONNX graph")] = 17,
    batch_size: Annotated[int, Parameter(help="Static batch size baked into the exported graph")] = 1,
    dynamic_batch: Annotated[
        bool,
        Parameter(help="Export with a dynamic batch dimension so the model accepts variable batch sizes at runtime"),
    ] = False,
    resolution: Annotated[
        Optional[int],
        Parameter(
            help=(
                "Input resolution (square side in pixels) baked into the graph. Defaults to the model's "
                "built-in resolution. Must be divisible by the backbone's patch_size * num_windows."
            )
        ),
    ] = None,
    backbone_only: Annotated[
        bool, Parameter(help="Export only the backbone (feature extractor) instead of the full detection model")
    ] = False,
    verbose: Annotated[bool, Parameter(help="Print export progress information")] = True,
):
    """Export an RF-DETR model archive to OpenVINO IR format.

    RF-DETR 1.8.1 does not export to OpenVINO directly, so the model is first exported to ONNX via RF-DETR's
    built-in exporter and then converted to OpenVINO IR (a ``.xml`` graph and its ``.bin`` weights) using
    ``openvino.convert_model``. The IR files are written to the experiment checkpoints folder and also copied to
    the experiment model folder so they are collected as model artifacts on the Hafnia platform.

    The export options mirror RF-DETR's ``export`` API: ``opset_version`` selects the ONNX opset used for the
    intermediate graph, ``batch_size`` bakes a static batch dimension into the graph (use ``dynamic_batch`` for a
    variable batch dimension instead), ``resolution`` overrides the square input size, and ``backbone_only``
    exports just the feature extractor.
    """
    logger = HafniaLogger(project_name="Export RF-DETR OpenVINO")

    # Prefer a user-selected checkpoint over the configured model when one is available.
    checkpoint_model_path = utils.get_checkpoint_if_available(logger)
    if checkpoint_model_path is not None:
        user_logger.info(f"Using checkpoint '{checkpoint_model_path.name}' instead of '{model_path}'")
        model_path = checkpoint_model_path.as_posix()

    # Load the model without 'optimize_for_inference' (no torch.compile), as ONNX export traces the
    # raw model.
    # The inference settings (InferenceConfig) is required by WrappedModel but not used during export.
    wrapped_model = WrappedModel.load_model(model_path, inference_config=InferenceConfig())

    # RF-DETR places the model on CUDA by default; fall back to CPU so export also works locally.
    if not torch.cuda.is_available():
        user_logger.info("CUDA is not available. Exporting on CPU.")
        wrapped_model.model.model.device = torch.device("cpu")

    output_dir = logger.path_model_checkpoints().as_posix()

    shape = (resolution, resolution) if resolution is not None else None

    configuration = {
        "model_filename": Path(model_path).name,
        "output_dir": output_dir,
        "opset_version": opset_version,
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "resolution": resolution,
        "backbone_only": backbone_only,
    }
    logger.log_configuration(configuration)

    # RF-DETR exports to ONNX first; the returned path points at the intermediate ONNX graph.
    onnx_path = Path(
        wrapped_model.model.export(
            output_dir=output_dir,
            opset_version=opset_version,
            batch_size=batch_size,
            dynamic_batch=dynamic_batch,
            shape=shape,
            backbone_only=backbone_only,
            verbose=verbose,
        )
    )

    # Convert the ONNX graph to OpenVINO IR, reusing the ONNX filename stem for the IR files.
    ov_model = ov.convert_model(onnx_path.as_posix())
    openvino_path = onnx_path.with_suffix(".xml")
    ov.save_model(ov_model, openvino_path.as_posix())
    user_logger.info(f"Exported OpenVINO model to '{openvino_path}'")

    # To store the model as both a checkpoint and a model artifact
    path_exported_models = logger.path_model()
    openvino_files = list(Path(output_dir).glob(f"{openvino_path.stem}.xml"))
    openvino_files += list(Path(output_dir).glob(f"{openvino_path.stem}.bin"))
    for exported_file in openvino_files:
        shutil.copy2(exported_file, path_exported_models)
        user_logger.info(f"Copied exported model to '{path_exported_models / exported_file.name}'")

    return logger


if __name__ == "__main__":
    # Creates launch schema file for the CLI function 'main'
    path_launch_schema = auto_save_command_builder_schema(main, cli_tool=utils.CLI_TOOL)
    user_logger.info(f"Launch schema saved to: {path_launch_schema}")

    app()
