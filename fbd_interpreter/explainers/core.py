import os

from fbd_interpreter.config.load import configuration
from fbd_interpreter.data_factory.resource.data_loader import (
    load_csv_resource,
    load_parquet_resource,
    load_pickle_resource,
)
from fbd_interpreter.explainers.shap_tree_explainer import ShapTreeExplainer
from fbd_interpreter.icecream import icecream
from fbd_interpreter.logger import logger
from fbd_interpreter.visualization.plots import plotly_figures_to_html

# Get html sections path
html_sections = configuration["DEV"]["html_sections"]


class Interpreter:
    def __init__(
        self,
        model_path,
        task_name="classification",
        tree_based_model=True,
        features_name=None,
        features_to_interpret=None,
        target_col=None,
        out_path=None,
    ):
        self.model = load_pickle_resource(model_path)
        self.features_name = features_name.split(",")
        self.features_to_interpret = features_to_interpret.split(",")
        self.target_col = target_col
        self.task_name = task_name
        self.tree_based_model = tree_based_model
        self.out_path = out_path
        self.out_path_global = os.path.join(out_path, "global_interpretation")
        self.out_path_local = os.path.join(out_path, "local_interpretation")

    def global_pdp_ice(self, train_data):
        classif = False
        if self.task_name == "classification":
            classif = True
        logger.info("Computing PDP & ice")
        pdp_plots = icecream.IceCream(
            data=train_data.drop([self.target_col], axis=1),
            feature_names=self.features_to_interpret,
            bins=10,
            model=self.model,
            targets=train_data[self.target_col],
            use_classif_proba=classif,
        )
        figs_pdp = pdp_plots.draw(kind="pdp", show=False)
        figs_ice = pdp_plots.draw(kind="ice", show=False)
        logger.info(f"Saving PD plots in {self.out_path_global}")
        plotly_figures_to_html(
            dic_figs=figs_pdp,
            path=self.out_path_global + "/partial_dependency_plots.html",
            title="Partial dependency plots ",
            plot_type="PDP",
            html_sections=html_sections,
        )
        logger.info(f"Saving ICE plots in {self.out_path_global}")
        plotly_figures_to_html(
            dic_figs=figs_ice,
            path=self.out_path_global + "/ice_plots.html",
            title="Individual Conditional Expectation (ICE) plots ",
            plot_type="ICE",
            html_sections=html_sections,
        )

        return None

    def global_ale(self, train_data):
        classif = False
        if self.task_name == "classification":
            classif = True
        logger.info("Computing ALE")
        ale_plots = icecream.IceCream(
            data=train_data[self.features_name],
            feature_names=self.features_to_interpret,
            bins=10,
            model=self.model,
            targets=train_data[self.target_col],
            use_classif_proba=classif,
            use_ale=True,
        )
        figs_ale = ale_plots.draw(kind="ale", show=False)
        logger.info(f"Saving ALE plots in {self.out_path_global}")
        plotly_figures_to_html(
            dic_figs=figs_ale,
            path=self.out_path_global + "/accumulated_local_effects_plots.html",
            title="Accumulated Local Effects (ALE) plots ",
            plot_type="ALE",
            html_sections=html_sections,
        )

        return None

    def global_shap(self, train_data):
        classif = False
        if self.task_name == "classification":
            classif = True
        logger.info("Computing SHAP")
        shap_exp = ShapTreeExplainer(
            model=self.model, features_name=self.features_name,
        )
        # apply SHAP
        fig_1, fig_2 = shap_exp.global_explainer(train_data)
        dict_figs = {"Summary bar plot": fig_1, "Summary bee-swarm plot": fig_2}
        logger.info(f"Saving SHAP plots in {self.out_path_global}")
        plotly_figures_to_html(
            dic_figs=dict_figs,
            path=self.out_path_global + "/shapely_values.html",
            title="SHAP values ",
            plot_type="SHAP_GLOBAL",
            html_sections=html_sections,
        )

        return None

    def local_shap(self, test_data):
        classif = False
        if self.task_name == "classification":
            classif = True
        shap_exp = ShapTreeExplainer(
            model=self.model, features_name=self.features_name,
        )
        for j in range(0, len(test_data.head(10))):
            logger.info(
                f"Computing and saving SHAP individual plots for {j+1}th observation in {self.out_path_local}"
            )
            shap_exp.local_explainer(
                test_data, j, output_path=self.out_path_local, classif=classif
            )

        return None
