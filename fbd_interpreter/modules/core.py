import os
from pprint import pprint

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot

from fbd_interpreter.config.load import configuration
from fbd_interpreter.data_factory.resource.data_loader import (
    load_csv_resource,
    load_parquet_resource,
    load_pickle_resource,
)
from fbd_interpreter.modules.explainer import (
    apply_ale_plot,
    apply_pdp_ice_plot,
    apply_shap_plot,
    shap_plot,
)

# Get data path as csv / parquet
from fbd_interpreter.visualization.plots import plotly_figures_to_html

data_path = configuration["DEV"]["data_path"]
# Get model path as pickle
model_path = configuration["DEV"]["model_path"]
# Get features name as list
features_name = configuration["DEV"]["features_name"]
# Get target_col as string
target_col = configuration["DEV"]["target_col"]
# Get task name as str
task_name = configuration["DEV"]["task_name"]
# Get task name as str
target_col = configuration["DEV"]["target_col"]
# Get output path as str
out_path = configuration["DEV"]["output_path"]
# Get html sections path
html_sections = configuration["DEV"]["html_sections"]


class Interpreter:
    def __init__(
        self,
        model_path,
        data_path,
        interpretation_type="default",
        task_name="classification",
        features_name=None,
        target_col=None,
        out_path=None,
    ):
        self.model = load_pickle_resource(model_path)
        # TODO: specify type in conf
        if os.path.isdir(data_path):
            self.data = load_parquet_resource(data_path)
        else:
            self.data = load_csv_resource(data_path)
        self.features_name = features_name.split(",")
        self.target_col = target_col
        self.task_name = task_name
        self.out_path = out_path
        self.out_path_global = os.path.join(out_path, "global_interpretation")
        self.out_path_local = os.path.join(out_path, "local_interpretation")

    def global_pdp_ice(self):
        classif = False
        if self.task_name == "classification":
            classif = True

        fig_pdp, fig_ice = apply_pdp_ice_plot(
            data=self.data,
            model=self.model,
            feature_names=self.features_name,
            target_col=self.target_col,
            classif=classif,
        )
        plotly_figures_to_html(
            dic_figs=fig_pdp,
            path=self.out_path_global + "/partial_dependency_plots.html",
            title="Partial dependency plots ",
            plot_type="PDP",
            html_sections=html_sections,
        )
        plotly_figures_to_html(
            dic_figs=fig_ice,
            path=self.out_path_global + "/ice_plots.html",
            title="Individual Conditional Expectation (ICE) plots ",
            plot_type="ICE",
            html_sections=html_sections,
        )

        return None

    def global_ale(self):
        classif = False
        if self.task_name == "classification":
            classif = True

        fig_ale = apply_ale_plot(
            data=self.data,
            model=self.model,
            feature_names=self.features_name,
            target_col=self.target_col,
            classif=classif,
        )
        plotly_figures_to_html(
            dic_figs=fig_ale,
            path=self.out_path_global + "/accumulated_local_effects_plots.html",
            title="Accumulated Local Effects (ALE) plots ",
            plot_type="ALE",
            html_sections=html_sections,
        )

        return None

    def global_shap(self):
        raise NotImplementedError
        classif = False
        if self.task_name == "classification":
            classif = True
        # apply SHAP
        fig_list = apply_shap_plot(
            data=self.data, model=self.model, feature_names=self.features_name,
        )
        return None

    def intepreter_locally(self, X_test):

        shap_figs = []
        for obs_num in len(X_test):
            fig = shap_plot(obs_num, self.model, X_test)
            shap_figs.append(fig)

        return shap_figs
