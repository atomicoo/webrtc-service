import h5py

def load_embedding_h5file(file_path, keyword):
    with h5py.File(file_path, 'r') as f:
        if keyword in f:
            return f[keyword][:]
    return None

def save_embedding_h5file(file_path, keyword, embedding):
    with h5py.File(file_path, 'a') as f:
        if keyword in f:
            del f[keyword]
        f.create_dataset(keyword, data=embedding)