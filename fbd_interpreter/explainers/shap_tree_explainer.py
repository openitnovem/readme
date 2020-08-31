import matplotlib
import shap


class ShapTreeExplainer(object):
    """
    Allows to explain globally or locally a tree based model using Tree SHAP algorithms.
    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees.

    :Parameters:
        - model:
            A tree based model. Following models are supported: XGBoost, LightGBM, CatBoost, Pyspark & most
            tree-based models in scikit-learn.,
        - features_name (List[str]):
            List of features names used to train the model
    """

    def __init__(self, model, features_name):
        self.model = model
        self.features_name = features_name

    def global_explainer(self, train_data):
        """
        Create a SHAP feature importance plot and SHAP summary plot, colored by feature values.

        :Parameters:
            - `kind` (str = "pdp")
                Kind of plot to draw, possibilities are:

                - "pdp": draws a Partial Dependency Plot
                - "box": draws a box plot of predictions for each bin of features
                - "ice": draws a Individual Conditional Expectation plot
                - "ale": draws an Accumulated Local Effects plot

            - `show` (bool = True)
                Option to show the plots in notebook
            - `save_path` (Optional[str] = None)
                Path to directory to save the plots,
                directory is created if it does not exist
            - `ice_nb_lines` (int = 15)
                Number of lines to draw if kind="ice"
            - `ice_clustering_method` (str = "quantiles")
                Sampling or clustering method to compute the best lines to draw if kind="ice",
                available methods:

                - "kmeans": automatic clustering using KMeans to get representative lines
                - "quantiles": division of predictions in quantiles to get lines
                - "random": random selection of rows among predictions

        :Return:
            - `figures` (Dict[str, go.FigureWidget])
                Dictionary of generated plots,
                keys are feature names, values are Plotly objects
        """
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
