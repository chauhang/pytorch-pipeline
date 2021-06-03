#!/usr/bin/env/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Visualization Component Class."""
from pytorch_pipeline.components.base.base_component import BaseComponent
from pytorch_pipeline.components.visualization.executor import Executor
from pytorch_pipeline.types import standard_component_specs


class Visualization(BaseComponent):  # pylint: disable=R0903
    """Visualization Component Class."""

    def __init__(  # pylint: disable=R0913
        self,
        mlpipeline_ui_metadata=None,
        mlpipeline_metrics=None,
        confusion_matrix_dict=None,
        test_accuracy=None,
        markdown=None,
    ):
        """Initializes the Visualization component.

        Args:
            mlpipeline_ui_metadata : path to save ui metadata
            mlpipeline_metrics : metrics to be uploaded
            confusion_metrics_dict : dict for the confusion metrics
            test_accuracy : test accuracy of the model
            markdown : markdown dictionary
        """
        super(BaseComponent, self).__init__()  # pylint: disable=E1003

        input_dict = {
            standard_component_specs.VIZ_CONFUSION_MATRIX_DICT: confusion_matrix_dict,
            standard_component_specs.VIZ_TEST_ACCURACY: test_accuracy,
            standard_component_specs.VIZ_MARKDOWN: markdown,
        }

        output_dict = {}

        exec_properties = {
            standard_component_specs.VIZ_MLPIPELINE_UI_METADATA: mlpipeline_ui_metadata,
            standard_component_specs.VIZ_MLPIPELINE_METRICS: mlpipeline_metrics,
        }

        spec = standard_component_specs.VisualizationSpec()
        self._validate_spec(
            spec=spec,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
        )
        if markdown:
            self._validate_markdown_spec(spec=spec, markdown_dict=markdown)
        Executor().Do(
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
        )

        self.output_dict = output_dict

    def _validate_markdown_spec(
        self, spec: standard_component_specs, markdown_dict: dict
    ):
        for key in spec.MARKDOWN_DICT:
            if key not in markdown_dict:
                raise ValueError(f"Missing mandatory key - {key}")
            else:
                self._type_check(
                    actual_value=markdown_dict[key],
                    key=key,
                    spec_dict=spec.MARKDOWN_DICT,
                )
