# Bert - like attempts

## Current tasks
- [ ] add model saving 
- [ ] add loss savings
- [ ] limit the number of tests evaluations => speed-up the training
- [x] split the file on separate classes
- [ ] change model to one in `Embedding.py` (regularization, parallel evaluation)
- [ ] debug

## Execution:
1. Start file - `BertBased.py`
2. The line with data generation:

```python
generate_data(df_path, INPUT_SIZE, OUTPUT_SIZE)
```

It should be commented if the data is already generated (executes on CPU, takes ~1.5h)