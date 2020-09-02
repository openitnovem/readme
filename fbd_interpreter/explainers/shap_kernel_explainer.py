from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import shap


# WIP
class ShapKernelExplainer(object):
    """
    Allows to explain globally or locally any non tree based model using Kernel SHAP method.
    Kernel SHAP is a method that uses a special weighted linear regression to compute the importance
    of each feature. The computed importance values are Shapley values from game theory and also
    coefficients from a local linear regression.

    :Parameters:
        - model:
            Trained model to interpret
        - features_name (List[str]):
            List of features names used to train the model
    """

    def __init__(self, model, features_name):
        self.model = model
        self.features_name = features_name

    def global_explainer(
        self, train_data: pd.DataFrame, classif: bool
    ) -> List[plt.figure]:
        """
        Create a SHAP feature importance plot and SHAP summary plot colored by feature values using Kernel Explainer.

        :Parameters:
            - `train_data` (pd.DataFrame)
                Dataframe of model inputs, used to explain the model
            - `classif` (bool)
                True, if it's a classification problem else False

        :Return:
            - `shap_fig1, shap_fig2` (List[plt.figure])
                SHAP summary plots
       """
        train = train_data[self.features_name]
        train_summary = shap.kmeans(train, 10)
        if classif:
            model_func = self.model.predict_proba
        else:
            model_func = self.model.predict
        explainer = shap.KernelExplainer(model_func, train_summary)
        shap_values = explainer.shap_values(train)
        shap_fig1 = plt.figure()
        shap.summary_plot(shap_values, train, show=False)
        shap_fig2 = plt.figure()
        # TODO: handle shap_fig2 if not classif
        if classif:
            shap.summary_plot(shap_values[1], train, show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, test_data: pd.DataFrame, num_obs: int, classif: bool):
        """
        Computes and save SHAP force plot for a given observation in a pandas dataframe using Kernel Explainer.

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
        if classif:
            explainer = shap.KernelExplainer(
                self.model.predict_proba, test_data, link="logit"
            )
            shap_values = explainer.shap_values(test_data.iloc[num_obs])
            shap_fig = shap.force_plot(
                explainer.expected_value[1],
                shap_values[1],
                test_data.iloc[num_obs],
                link="logit",
            )
        else:
            explainer = shap.KernelExplainer(self.model.predict, test_data)
            shap_values = explainer.shap_values(test_data.iloc[num_obs])
            shap_fig = shap.force_plot(
                explainer.expected_value, shap_values, test_data.iloc[num_obs]
            )

        return shap_fig
