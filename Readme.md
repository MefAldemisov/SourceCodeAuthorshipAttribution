# Source Code Authorship attribution

## File organization

1. `inputs>processed_dfs` - datasets for actual training of the models, they are more light-weight, then original dataset, which is hidden by `.gitignore`
2. `src` - python and `.ipynb` files
   2.1 `data_processing` - work with direct GCJ (time consuming)
   2.2 `models` - main directory
   2.3 `training` - things, which are used by the models
   2.4 `visualization`
   2.5 `main.py` - **starting point**
3. `outputs` - images, models e.t.c

## Class diagram (TODO)
(simple description instead of it)

1. Model - root class (interface for all models)
2. Triplet(Model) - triplet-loss specific methods (batch generation, fit process, full model creation e.t.c)
3. Embedding(Triplet) - actual realization of the target architecture
