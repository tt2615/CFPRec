import pickle
import glob

city = 'NY'
outfile = open(f'./result_final/result_{city}.txt', 'a')
max_global_recall = {1:[('',0.,0)],5:[('',0.,0)],10:[('',0.,0)]}
max_global_ndcg = {1:[('',0.,0)],5:[('',0.,0)],10:[('',0.,0)]}
max_global_map = {1:[('',0.,0)],5:[('',0.,0)],10:[('',0.,0)]}

for i, file_name in enumerate(glob.glob(f'./result_{city}/*')):
    print(i)
    file = open(file_name, 'rb')
    for i in range(4):
        data = pickle.load(file)
        if i == 1: #recall
            max_local_recall = {1:(0,0.),5:(0,0.),10:(0,0.)}
            for epoch, recalls in data.items():
                for k, recall in recalls.items():
                    recall = recall.item()
                    if max_local_recall[k][1] < recall:
                        max_local_recall[k] = (epoch, recall)
                    if max_global_recall[k][0][1] < recall:
                        max_global_recall[k] = [(file_name, recall, epoch)]
                    elif max_global_recall[k][0][1] == recall:
                        max_global_recall[k].append((file_name, recall, epoch))

        elif i == 2: #ndcg
            max_local_ndcg = {1:(0,0.),5:(0,0.),10:(0,0.)}
            for epoch, ndcgs in data.items():
                for k, ndcg in ndcgs.items():
                    ndcg = ndcg.item()
                    if max_local_ndcg[k][1] < ndcg:
                        max_local_ndcg[k] = (epoch, ndcg)
                    if max_global_ndcg[k][0][1] < ndcg:
                        max_global_ndcg[k] = [(file_name, ndcg, epoch)]
                    elif max_global_ndcg[k][0][1] == ndcg:
                        max_global_ndcg[k].append((file_name, ndcg, epoch))
        elif i == 3: #map
            max_local_map = {1:(0,0.),5:(0,0.),10:(0,0.)}
            for epoch, maps in data.items():
                for k, map in maps.items():
                    map = map.item()
                    if max_local_map[k][1] < map:
                        max_local_map[k] = (epoch, map)
                    if max_global_map[k][0][1] < map:
                        max_global_map[k] = [(file_name, map, epoch)]
                    elif max_global_map[k][0][1] == map:
                        max_global_map[k].append((file_name, map, epoch))
    file.close()

    outfile.write(f"{file_name}\n")
    outfile.write(f"recall: {max_local_recall}\n")
    outfile.write(f"ndcg: {max_local_ndcg}\n")
    outfile.write(f"map: {max_local_map}\n")
    outfile.write('--------------\n\n')

outfile.write('max global recall:\n')
for k, list in max_global_recall.items():
    outfile.write(f"{k}: {list}\n")
outfile.write('max global ndcg:\n')
for k, list in max_global_ndcg.items():
    outfile.write(f"{k}: {list}\n")
outfile.write('max global map:\n')
for k, list in max_global_map.items():
    outfile.write(f"{k}: {list}\n")
outfile.write("**************\n")

outfile.close()
