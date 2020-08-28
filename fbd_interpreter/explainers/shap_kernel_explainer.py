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

    def __init__(self, model, train, test, features_name):
        self.model = model
        self.train = train[features_name]
        self.train_summary = shap.kmeans(self.train, 10)
        self.test = test[features_name]

    def global_explainer(self, classif):
        if classif:
            model_func = self.model.predict_proba
        else:
            model_func = self.model.predict
        explainer = shap.KernelExplainer(model_func, self.train_summary)
        shap_values = explainer.shap_values(self.train)
        shap_fig1 = matplotlib.pyplot.figure()
        shap.summary_plot(shap_values, self.train, show=False)
        shap_fig2 = matplotlib.pyplot.figure()
        if classif:
            shap.summary_plot(shap_values[1], self.train, show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, j, classif, output_path):
        if classif:
            explainer = shap.KernelExplainer(
                self.model.predict_proba, self.test, link="logit"
            )
            shap_values = explainer.shap_values(self.test.iloc[j])
            shap.save_html(
                output_path + f"/shap_local_kernel_explanation_{j}_th_obs.html",
                shap.force_plot(
                    explainer.expected_value[1],
                    shap_values[1],
                    self.test.iloc[j],
                    link="logit",
                ),
            )
        else:
            explainer = shap.KernelExplainer(self.model.predict, self.test)
            shap_values = explainer.shap_values(self.test.iloc[j])
            shap.save_html(
                output_path + f"/shap_local_kernel_explanation_{j}_th_obs.html",
                shap.force_plot(
                    explainer.expected_value, shap_values, self.test.iloc[j]
                ),
            )

        return None
