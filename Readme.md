# Source Code Authorship attribution

## File organization

1. `inputs>processed_dfs` - datasets for actual training of the models, they are more light-weight, then original dataset, which is hidden by `.gitignore`
2. `src` - python and `.ipynb` files
   
      2.1 `data_processing/` - work with direct GCJ (time-consuming)
   
      2.2 `models/` - main directory with the models with the inheritance hierarchy, described below
   
      2.3 `training/` - things, which are used by the models: `GridSearch` methods and `Training callback`
   
      2.4 `main.py` - **starting point**
   
3. `outputs` - images, models e.t.c


## How to run

1. Install the requirements from `requiremnets.txt`:
``pip install -r requirements.txt``
2. Change the `src/main.py` as appropriate:
   (uncomment the following lines, if needed):
   
2.1 To train the embedding-based model:
``` python 
embedding = Embedding(make_initial_preprocess=False)
embedding.train(batch_size=128, epochs=1)
```
This commands will create the model and train it for one epoch with the batch size 128. 
The dataset will be taken from `inputs/preprocessed_jsons/embedding_train.json`, 
if `make_initial_preprocess` is set to `False`. Otherwise, the access to the raw data is required.

2.2 To train the embedding-based model:
```python
conv2d = Conv2D(make_initial_preprocess=True)
conv2d.train(batch_size=128, epochs=1)
```
**Warning:** to train this model, the row dataset(`py_df.csv`) is required

2.3 To generate the images, which represent the focus of the models:
```python
Visualizer("conv2d").run()
Visualizer("embedding").run()
```

2.4 To show all the layers of the models:
```python
KeractVisualizer("conv2d").run()
KeractVisualizer("embedding").run()
```

3. Run `python3 src/main.py` and fix import errors, if there are

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
- [Good description of the visualization domain](https://medium.com/google-developer-experts/interpreting-deep-learning-models-for-computer-vision-f95683e23c1d)
