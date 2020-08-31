import matplotlib
import shap


class ShapTreeExplainer(object):
    """Uses Tree SHAP algorithms to explain the output of ensemble tree models.

        Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees,
        under several different possible assumptions about feature dependence. It depends on fast C++
        implementations either inside an externel model package or in the local compiled C extention.

        Parameters
        ----------
        model : model object
        """

    def __init__(self, model, features_name):
        """

        :param model: A tree based model. Following models are supported by Tree SHAP at present: XGBoost, LightGBM,
                    CatBoost, Pyspark & most tree-based models in scikit-learn.
        :type model:
        :param features_name:
        :type features_name:
        """
        self.model = model
        self.features_name = features_name

    def global_explainer(self, train_data):
        shap_values = shap.TreeExplainer(self.model).shap_values(
            train_data[self.features_name]
        )
        shap_fig1 = matplotlib.pyplot.figure()
        shap.summary_plot(
            shap_values, train_data[self.features_name], plot_type="bar", show=False
        )
        shap_fig2 = matplotlib.pyplot.figure()
        shap.summary_plot(shap_values, train_data[self.features_name], show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, test_data, num_obs, classif, output_path):
        test_data = test_data[self.features_name]
        explainerModel = shap.TreeExplainer(self.model)
        shap_values_Model = explainerModel.shap_values(test_data.iloc[num_obs])
        if classif:
            shap.save_html(
                output_path + f"/shap_local_explanation_{num_obs + 1}th_obs.html",
                shap.force_plot(
                    explainerModel.expected_value,
                    shap_values_Model,
                    test_data.iloc[num_obs],
                    link="logit",
                ),
            )
        else:
            shap.save_html(
                output_path + f"/shap_local_explanation_{num_obs + 1}th_obs.html",
                shap.force_plot(
                    explainerModel.expected_value,
                    shap_values_Model,
                    test_data.iloc[num_obs],
                ),
            )
        return None
