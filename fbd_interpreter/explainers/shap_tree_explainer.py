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
            The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost, Pyspark
            and most tree-based scikit-learn models are supported.

        data : numpy.array or pandas.DataFrame
            The background dataset to use for integrating out features. This argument is optional when
            feature_perturbation="tree_path_dependent", since in that case we can use the number of training
            samples that went down each tree path as our background dataset (this is recorded in the model object).

        feature_perturbation : "interventional" (default) or "tree_path_dependent" (default when data=None)
            Since SHAP values rely on conditional expectations we need to decide how to handle correlated
            (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between
            features according to the rules dictated by casual inference (Janzing et al. 2019). Note that the
            "interventional" option requires a background dataset and its runtime scales linearly with the size
            of the background dataset you use. Anywhere from 100 to 1000 random background samples are good
            sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number
            of training examples that went down each leaf to represent the background distribution. This approach
            does not require a background dataset and so is used by default when no background dataset is provided.

        model_output : "raw", "probability", "log_loss", or model method name
            What output of the model should be explained. If "raw" then we explain the raw output of the
            trees, which varies by model. For regression models "raw" is the standard output, for binary
            classification in XGBoost this is the log odds ratio. If model_output is the name of a supported
            prediction method on the model object then we explain the output of that model method name.
            For example model_output="predict_proba" explains the result of calling model.predict_proba.
            If "probability" then we explain the output of the model transformed into probability space
            (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
            then we explain the log base e of the model loss function, so that the SHAP values sum up to the
            log loss of the model for each sample. This is helpful for breaking down model performance by feature.
            Currently the probability and logloss options are only supported when feature_dependence="independent".
        """

    def __init__(self, model, train, test, features_name):
        """

        :param model: A tree based model. Following models are supported by Tree SHAP at present: XGBoost, LightGBM,
                    CatBoost, Pyspark & most tree-based models in scikit-learn.
        :type model:
        :param train:
        :type train:
        :param test:
        :type test:
        :param features_name:
        :type features_name:
        """
        self.model = model
        self.train = train[features_name]
        self.test = test[features_name]
        self.shap_values = None

    def global_explainer(self):
        shap_values = shap.TreeExplainer(self.model).shap_values(self.train)
        shap_fig1 = matplotlib.pyplot.figure()
        shap.summary_plot(shap_values, self.train, plot_type="bar", show=False)
        shap_fig2 = matplotlib.pyplot.figure()
        shap.summary_plot(shap_values, self.train, show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, j, output_path, classif):
        explainerModel = shap.TreeExplainer(self.model)
        shap_values_Model = explainerModel.shap_values(self.test.iloc[j])
        if classif:
            shap.save_html(
                output_path + f"/shap_local_explanation_{j+1}th_obs.html",
                shap.force_plot(
                    explainerModel.expected_value,
                    shap_values_Model,
                    self.test.iloc[j],
                    link="logit",
                ),
            )
        else:
            shap.save_html(
                output_path + f"/shap_local_explanation_{j+1}th_obs.html",
                shap.force_plot(
                    explainerModel.expected_value, shap_values_Model, self.test.iloc[j],
                ),
            )
        return None
