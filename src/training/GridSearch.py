import tqdm
from models.Model import Model

def generate_param_list(params: dict):
    param_list = []
    for key in params.keys():
        if len(param_list) == 0:
            for x in params[key]:
                param_list.append([x])
        else:
            new_param_list = []
            for l in param_list:
                for x in params[key]:
                    l_copy = l.copy()
                    l_copy.append(x)
                    new_param_list.append(l_copy)
            param_list = new_param_list
    return param_list

# def grid_search(params,):
#     param_list = generate_param_list(params)
#     max_score = 0
#     best_model = None
#     log = []
#     core_model = None
#     for p in tqdm.tqdm(param_list[3:4]):
#         core_model = create_model(*p)
#         triplet = create_triplet_model(input_size, core_model)
#         model, history = train(new_X, new_y, input_size, triplet)
#         score = test(core_model, new_X, new_y)
#         if score > max_score:
#             max_score = score
#             best_model = model
#             print(p, score)
#         log.append({"score": score, "history": history, "params": p})