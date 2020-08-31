import matplotlib
import shap


# WIP
class ShapKernelExplainer(object):
    """Uses the Kernel SHAP method to explain the output of any function.

        Kernel SHAP is a method that uses a special weighted linear regression
        to compute the importance of each feature. The computed importance values
        are Shapley values from game theory and also coefficents from a local linear
        regression.


        Parameters
        ----------
        model : function or iml.Model
            Trained model to interpret

        train : pandas.DataFrame
            Train dataframe - used for global interpretation

        test : pandas.DataFrame
            Test dataframe or dataframe that contains observations to explain - used for local explanation

        features_name : List of string
            List of features names
        """

    def __init__(self, model, features_name):
        self.model = model
        self.features_name = features_name

    def global_explainer(self, train_data, classif):
        train = train_data[self.features_name]
        train_summary = shap.kmeans(train, 10)
        if classif:
            model_func = self.model.predict_proba
        else:
            model_func = self.model.predict
        explainer = shap.KernelExplainer(model_func, train_summary)
        shap_values = explainer.shap_values(train)
        shap_fig1 = matplotlib.pyplot.figure()
        shap.summary_plot(shap_values, train, show=False)
        shap_fig2 = matplotlib.pyplot.figure()
        # TODO: handle shap_fig2 if not classif
        if classif:
            shap.summary_plot(shap_values[1], train, show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, test_data, num_obs, classif, output_path):
        test_data = test_data[self.features_name]
        if classif:
            explainer = shap.KernelExplainer(
                self.model.predict_proba, test_data, link="logit"
            )
            shap_values = explainer.shap_values(test_data.iloc[num_obs])
            shap.save_html(
                output_path + f"/shap_local_kernel_explanation_{num_obs}_th_obs.html",
                shap.force_plot(
                    explainer.expected_value[1],
                    shap_values[1],
                    test_data.iloc[num_obs],
                    link="logit",
                ),
            )
        else:
            explainer = shap.KernelExplainer(self.model.predict, test_data)
            shap_values = explainer.shap_values(test_data.iloc[num_obs])
            shap.save_html(
                output_path + f"/shap_local_kernel_explanation_{num_obs}_th_obs.html",
                shap.force_plot(
                    explainer.expected_value, shap_values, test_data.iloc[num_obs]
                ),
            )

        return None
