import os

def model_directory(model_path):
    if "their_split_all_processes_mixed" in model_path:
        return "_all_processes_mixed_their_split"
    elif "their_split_all_processes" in model_path:
        return "_all_processes_their_split"
    elif "their_split_mixed" in model_path:
        return "_mixed_their_split"
    elif "their_split_WORD" in model_path:
        return "_their_split_WORD"
    elif "their_split" in model_path:
        return "_their_split"

def find_files(model_path):
    train, dev, test = "","",""
    directory_files = model_path.split("/")
    if len(directory_files)>2:
      directory_files = "their_split_all_processes"
    else:
      directory_files = directory_files[-1]
    path_dir = os.path.join("files", directory_files)
    files = [os.path.join(path_dir, f) for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))]   
    for f in files:
      if "train" in f:
        train = f
      elif "dev" in f:
        dev = f
      elif "test" in f:
        test=f
    return train,dev,test 
    