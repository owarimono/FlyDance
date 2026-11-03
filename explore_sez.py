import pickle

x = pickle.load(open('data/sez_neurons.pickle', 'rb'))

print('Type:', type(x))
print('Keys count:', len(x))
print('All keys:', list(x.keys()))
print()

# Show first 5 values in detail
for k in list(x.keys())[:5]:
    v = x[k]
    print('---')
    print('Key:', k)
    print('Value type:', type(v))
    print('Value length:', len(v) if hasattr(v, '__len__') else 'N/A')
    print('Value preview:', str(v)[:300])
    print()