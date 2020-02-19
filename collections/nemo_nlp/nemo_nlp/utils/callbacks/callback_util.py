
def get_output(tensors, global_vars, dict_keys, to_cpu=False, delim='~~~'):
    for key in dict_keys:
        if key not in global_vars:
            global_vars[key] = []

    output = {}
    for k, v in tensors.items():
        ind = k.find(delim)
        if ind != -1:
            output[k[:ind]] = v[0]

    return global_vars, output

    # for evary var to check if it needs to be moved to cpu or not
