import pickle

file_path = "/root/amour-fusion/data/BTXRD/notes_encoded.p"

with open(file_path, "rb") as f:
    data = pickle.load(f)

# Check what type of data it is and print the first 5 items
print(f"Data type: {type(data)}")

if isinstance(data, (list, tuple)):
    print("First 5 items:")
    print(data[:5])
elif isinstance(data, dict):
    # Print first 5 keys and their values
    print("First 5 key-value pairs:")
    first_5 = {k: data[k] for k in list(data.keys())[:5]}
    print(first_5)
else:
    # If it's a pandas DataFrame or other object
    print("Data head:")
    print(data.head(5) if hasattr(data, 'head') else data)