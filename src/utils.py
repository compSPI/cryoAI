import os


def cond_mkdir(path):
    if not os.path.exists(path): # path does not exist, create it
        os.makedirs(path)
        return True

    # else:
    #     val = input(f"Directory {path} exists, do you want to overwrite it [Y]/n? ")
    #     if val == '' or val == 'Y': # path exists, overwrite
    #         shutil.rmtree(path)
    #         os.makedirs(path)
    #         return True
    #     else:   # path exists, return
    #         return False
    return True
