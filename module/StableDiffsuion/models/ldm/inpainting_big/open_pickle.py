import pickle


with open('archive/data.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)