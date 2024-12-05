{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e69eef57-638a-4d99-a755-faa36a7bf69a",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'shap'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mshap\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mmatplotlib\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpyplot\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mplt\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mexplain_prediction\u001b[39m(model, input_data, feature_names):\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'shap'"
     ]
    }
   ],
   "source": [
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def explain_prediction(model, input_data, feature_names):\n",
    "    explainer = shap.TreeExplainer(model)\n",
    "    shap_values = explainer.shap_values(input_data)\n",
    "\n",
    "    shap.initjs()\n",
    "    return shap.force_plot(\n",
    "        explainer.expected_value[1], shap_values[1], input_data, feature_names=feature_names\n",
    "    )\n",
    "\n",
    "def summary_plot(model, X_test):\n",
    "    explainer = shap.TreeExplainer(model)\n",
    "    shap_values = explainer.shap_values(X_test)\n",
    "\n",
    "    shap.summary_plot(shap_values[1], X_test, plot_type=\"bar\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
