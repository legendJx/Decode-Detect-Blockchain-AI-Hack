# What we did not write ourselves

- **Starter notebook and `payments.py`**: the organiser-supplied starter files from the
  Decode & Detect training download. `detector.ipynb` is built on the starter notebook; the
  data loading and baseline features are the organisers' code.
- **Libraries**: NumPy, pandas, PyArrow, scikit-learn, LightGBM, NetworkX, Matplotlib,
  Jupyter/nbconvert/ipykernel (see `requirements.txt`), and the OpenMP runtime (libomp) that
  LightGBM uses on macOS.
- **Pre-trained models / outside services**: none.
- **AI assistance**: Claude Code (Anthropic) was used to analyse the training data, write the
  fast pass-through and scenario-volume features, switch the model to LightGBM and choose its
  settings, add the scenario-grouped check, extend the flow-plot cell, run the notebook, check
  the predictions file and prepare this package (README, requirements, licence, this file).
