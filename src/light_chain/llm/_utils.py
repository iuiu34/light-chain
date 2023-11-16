def index_list(x):
    out = [f"{k + 1}. {v}" for k, v in enumerate(x)]
    out = '\n'.join(out)
    return out