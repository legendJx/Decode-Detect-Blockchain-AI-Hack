# What we did not write ourselves

- **Organiser files**: the data-format description in the Decode & Detect README guided the
  parsing. `payments.py` and the starter notebook are not used by this version.
- **Libraries**: NumPy, pandas, PyArrow, SciPy, scikit-learn, LightGBM, NetworkX, Matplotlib,
  Jupyter/nbconvert/nbformat/ipykernel (see `requirements.txt`), and the OpenMP runtime
  (libomp) that LightGBM uses on macOS.
- **Methods from the literature**: focal loss (Lin et al., 2017), used as one of the compared
  loss functions; SHAP-style feature contributions via LightGBM's built-in `pred_contrib`.
- **Pre-trained models / outside services**: none.
- **AI assistance**: Claude Code (Anthropic) was used to analyse the training data and derive
  the base rules, and to write `aml_features.py` and the notebook: feature matrix, scenario-level
  aggregates, the v10 account-fingerprint removal and robust core blend, the v9 blob fix and shape-relative medium features, amount-matched flows, hyperparameter search, loss comparison, loss × seed ensemble,
  plain-language plots and summary. It also ran the notebook, checked the predictions file and
  prepared this package (README, requirements, this file).
