# Source Code Authorship attribution

## File organization

1. `inputs>processed_dfs` - datasets for actual training of the models, they are more light-weight, then original dataset, which is hidden by `.gitignore`
2. `src` - python and `.ipynb` files
   
      2.1 `data_processing/` - work with direct GCJ (time-consuming)
   
      2.2 `models/` - main directory with the models with the inheritance hierarchy, described below
   
      2.3 `training/` - things, which are used by the models: `GridSearch` methods and `Training callback`
   
      2.4 `main.py` - **starting point**
   
3. `outputs` - images, models e.t.c

## Class diagram (TODO)
(simple description instead of it)

1. `Model` - root class (interface for all models)
2. `Triplet(Model)` - triplet-loss specific methods (batch generation, fit process, full model creation e.t.c)
3. `Embedding(Triplet)` - actual realization of the target architecture

## Visualization:

1. `Visualizer` - visualization, based on [`tf-keras-vis`](https://github.com/keisen/tf-keras-vis), which performs per-pixel modifications of the image, which potentially can lead to errors
2. `KeractVisualizer` - visualization, based on [`keract`](https://github.com/philipperemy/keract) library. 
   
**WARNING:** when using, substitute the `._layers` call with `.layers` call in `keract.py` file within a library in case of error (tensorflow version 2.5.0, [keract](https://github.com/philipperemy/keract) version 4.4.0) 

## Useful links
- [Tensorflow addons, hard triplet loss](https://github.com/tensorflow/addons/blob/30c8a7094f3bdcca5cc26fc88c1e33f022782266/tensorflow_addons/losses/triplet.py#L204)