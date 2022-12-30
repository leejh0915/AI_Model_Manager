import importlib

def load_module_func(module_name):
    import_info = module_name.rsplit('.', 1)
    res = getattr(importlib.import_module(import_info[0]), import_info [1])
    return res