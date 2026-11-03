import pickle

x = pickle.load(open('data/sez_neurons.pickle', 'rb'))

# Categorize by name pattern
categories = {
    'Descending (aDT/DN)': [],
    'Sensory (SG/FMIn)': [],
    'Motor (MN)': [],
    'Other': [],
}

for name in sorted(x.keys()):
    n = len(x[name])
    if name.startswith(('aDT', 'DN')):
        categories['Descending (aDT/DN)'].append((name, n))
    elif 'SG' in name or name.startswith('FMIn'):
        categories['Sensory (SG/FMIn)'].append((name, n))
    elif 'MN' in name:
        categories['Motor (MN)'].append((name, n))
    else:
        categories['Other'].append((name, n))

for cat, items in categories.items():
    total = sum(n for _, n in items)
    print(f'\n=== {cat} ({len(items)} types, {total} neurons) ===')
    for name, n in items:
        print(f'  {name:25s} : {n}')