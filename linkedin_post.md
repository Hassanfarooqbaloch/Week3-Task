🚀 Week 3 of my Data Science Internship: PCA, Deployment, and a Live Dashboard

This week I took last week's best model — a Random Forest trained on steel plant sensor data —
through the two things that come right before and right after "the model works": dimensionality
reduction and deployment.

📉 Part 1: Does PCA actually help here?
• Recreated last week's exact 20-feature set and train/test split, then fit StandardScaler +
  PCA on the training data only (fitting on the full dataset first would leak test info into
  the transformation — an easy mistake to make)
• The cumulative variance curve showed 10 of 20 components capture 95% of variance
• Retrained the model on just 3 components, and separately on the 95%-variance set, to see the
  real cost of compression
• Result: RMSE got ~7x worse with 3 components, and still ~3x worse even at 95% retained
  variance. Random Forest already does its own feature selection through splits, so PCA was
  mostly throwing away information the trees could use directly — a good reminder that PCA
  isn't a free win for every model type

🖥️ Part 2: Shipping it
• Built a FastAPI app with three routes: a home page, a dashboard showing the Week 2 EDA charts,
  and a live prediction form
• The prediction form is generated dynamically from the model's own metadata, so it always
  matches whatever the model actually expects
• Ran into a real deployment lesson here too: the unconstrained Random Forest saved to ~380MB —
  too big for GitHub. Capping tree depth brought it down to ~18MB with only a small accuracy
  trade-off, which mattered more for actually shipping this than squeezing out the last bit of
  test-set RMSE

📸 [Insert screenshot of the running dashboard here]

💡 Biggest takeaway: a model isn't "done" when it hits a good RMSE. Whether it can actually be
packaged, served, and used by someone else is its own set of decisions — and this week made
that very concrete.

Full code, the PCA notebook, and the FastAPI app are on GitHub 👉 [link]

#DataScience #MachineLearning #FastAPI #PCA #MLEngineering #Internship #Python
