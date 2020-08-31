import os
from typing import Any, List

from fbd_interpreter.config.load import configuration
from fbd_interpreter.explainers.shap_kernel_explainer import ShapKernelExplainer
from fbd_interpreter.explainers.shap_tree_explainer import ShapTreeExplainer
from fbd_interpreter.icecream import icecream
from fbd_interpreter.logger import logger
from fbd_interpreter.visualization.plots import plotly_figures_to_html

# Get html sections path
html_sections = configuration["DEV"]["html_sections"]


class Interpreter:
    """
    Class that contains different interpretability techniques to explain the model training (global interpretation)
    and its predictions (local interpretation).
    :Parameters:
        - model:
            Model to compute predictions using provided data,
            `model.predict(data)` must work
        - task_name (str):
            Task name: choose from supported_tasks in config/config_{type_env}.cfg
        - tree_based_model (str):
            If "True", we use Tree SHAP algorithms to explain the output of ensemble tree models.
        - features_name (List[str]):
            List of features names used to train the model
        - features_to_interpret (List[str]):
            List of features to interpret using pdp, ice and ale
        - target_col (str):
            name of target column
        - out_path (str):
            Output path used to save interpretability plots.
    :Return:
        None
    """

    def __init__(
        self,
        model: Any,
        task_name: str = "classification",
        tree_based_model: str = None,
        features_name: List[str] = None,
        features_to_interpret: List[str] = None,
        target_col: str = None,
        out_path: str = None,
    ):

        self.model = model
        self.features_name = features_name
        self.features_to_interpret = features_to_interpret
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
            path=self.out_path_global
            + "/individual_conditional_expectation_plots.html",
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
        if self.tree_based_model == "True":
            logger.info(
                "You are using a tree based model, if it's not the case, please set tree_based_model to False in config/config_{type_env}.cfg"
            )

            shap_exp = ShapTreeExplainer(
                model=self.model, features_name=self.features_name,
            )
            # apply SHAP
            shap_fig_1, shap_fig_2 = shap_exp.global_explainer(train_data)
        elif self.tree_based_model == "False":
            logger.info(
                "You are using a non tree based model, if it's not the case, please set tree_based_model to True in config/config_{type_env}.cfg"
            )
            shap_exp = ShapKernelExplainer(
                model=self.model, features_name=self.features_name,
            )
            # apply SHAP
            shap_fig_1, shap_fig_2 = shap_exp.global_explainer(
                train_data, classif=classif
            )
        else:
            logger.error(
                "Please set tree_based_model to True or False in config/config_{type_env}.cfg"
            )

        dict_figs = {
            "Summary bar plot": shap_fig_1,
            "Summary bee-swarm plot": shap_fig_2,
        }
        logger.info(f"Saving SHAP plots in {self.out_path_global}")
        plotly_figures_to_html(
            dic_figs=dict_figs,
            path=self.out_path_global + "/shap_feature_importance_plots.html",
            title="SHAP feature importance plots",
            plot_type="SHAP_GLOBAL",
            html_sections=html_sections,
        )

        return None

    def local_shap(self, test_data):
        classif = False
        if self.task_name == "classification":
            classif = True

        if self.tree_based_model == "True":
            logger.info(
                "You are using a tree based model, if it's not the case, please set tree_based_model to False in config/config_{type_env}.cfg"
            )

            shap_exp = ShapTreeExplainer(
                model=self.model, features_name=self.features_name,
            )

        elif self.tree_based_model == "False":
            logger.info(
                "You are using a non tree based model, if it's not the case, please set tree_based_model to True in config/config_{type_env}.cfg"
            )
            shap_exp = ShapKernelExplainer(
                model=self.model, features_name=self.features_name,
            )

        else:
            logger.error(
                "Please set tree_based_model to True or False in config/config_{type_env}.cfg"
            )

        for j in range(0, len(test_data.head(10))):
            logger.info(
                f"Computing and saving SHAP individual plots for {j + 1}th observation in {self.out_path_local}"
            )
            shap_exp.local_explainer(
                test_data=test_data,
                num_obs=j,
                classif=classif,
                output_path=self.out_path_local,
            )
        return None
