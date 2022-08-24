import os 

path = '/home/toefl/K/nto/checkpoints/stackmix'

files = os.listdir(path)
tosave = sorted(files, key=lambda x: int(x.split('-')[1]) if x != 'log.txt' else 0)
tosave = ['log.txt', tosave[-1]]

for file in files:
    if file not in tosave:
        os.remove(os.path.join(path, file))
