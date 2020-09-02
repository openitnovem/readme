from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import shap


class ShapTreeExplainer(object):
    """
    Allows to explain globally or locally a tree based model using Tree SHAP algorithms.
    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees.

    :Parameters:
        - model:
            A tree based model. Following models are supported: XGBoost, LightGBM, CatBoost, Pyspark & most
            tree-based models in scikit-learn...
        - features_name (List[str]):
            List of features names used to train the model
    """

    def __init__(self, model, features_name):
        self.model = model
        self.features_name = features_name

    def global_explainer(self, train_data: pd.DataFrame) -> List[plt.figure]:
        """
        Create a SHAP feature importance plot and SHAP summary plot, colored by feature values using Tree Explainer.

        :Parameters:
            - `train_data` (pd.DataFrame)
                Dataframe of model inputs, used to explain the model

        :Return:
            - `shap_fig1, shap_fig2` (List[plt.figure])
                SHAP summary plots
        """
        shap_values = shap.TreeExplainer(self.model).shap_values(
            train_data[self.features_name]
        )
        shap_fig1 = plt.figure()
        shap.summary_plot(
            shap_values, train_data[self.features_name], plot_type="bar", show=False
        )
        shap_fig2 = plt.figure()
        shap.summary_plot(shap_values, train_data[self.features_name], show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, test_data: pd.DataFrame, num_obs: int, classif: bool):
        """
        Computes and save SHAP force plot for a given observation in a pandas dataframe using Tree Explainer.

        :Parameters:
            - `test_data` (pd.DataFrame)
                Dataframe of observations to interpret, must have the same features as the model inputs
            - `num_obs` (int)
                The observation number to explain (n° raw in test_data)
            - `classif` (bool)
                True, if it's a classification problem, else False
            - `output_path` (str)
                Output path used to save plots.
        """
        test_data = test_data[self.features_name]
        explainerModel = shap.TreeExplainer(self.model)
        shap_values_Model = explainerModel.shap_values(test_data.iloc[num_obs])
        if classif:
            shap_fig = shap.force_plot(
                explainerModel.expected_value,
                shap_values_Model,
                test_data.iloc[num_obs],
                link="logit",
            )
        else:
            shap_fig = shap.force_plot(
                explainerModel.expected_value,
                shap_values_Model,
                test_data.iloc[num_obs],
            )
        return shap_fig
