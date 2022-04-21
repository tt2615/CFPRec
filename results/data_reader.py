import pickle


city = "PHO"

file_name = f"./{city}_log"
file = open(file_name, 'rb')

outfile = open(f"./result_{city}.txt", 'a')

epoch_data = {epoch:{1:{}, 5:{}, 10:{}} for epoch in range(100)}

for i in range(4): 
    data = pickle.load(file)
    if i == 1: #recall
        max_local_recall = {1:(0,0.),5:(0,0.),10:(0,0.)}
        for epoch, recalls in data.items():
            for k, recall in recalls.items():
                recall = recall.item()
                epoch_data[epoch][k]['recall'] = recall
                if max_local_recall[k][1] < recall:
                    max_local_recall[k] = (epoch, recall)

    elif i == 2: #ndcg
        max_local_ndcg = {1:(0,0.),5:(0,0.),10:(0,0.)}
        for epoch, ndcgs in data.items():
            for k, ndcg in ndcgs.items():
                ndcg = ndcg.item()
                epoch_data[epoch][k]['ndcg'] = ndcg
                if max_local_ndcg[k][1] < ndcg:
                    max_local_ndcg[k] = (epoch, ndcg)
    elif i == 3: #map
        max_local_map = {1:(0,0.),5:(0,0.),10:(0,0.)}
        for epoch, maps in data.items():
            for k, map in maps.items():
                map = map.item()
                epoch_data[epoch][k]['map'] = map
                if max_local_map[k][1] < map:
                    max_local_map[k] = (epoch, map)

outfile.write(f"{file_name}\n")
outfile.write(f"recall: {max_local_recall}\n")
outfile.write(f"ndcg: {max_local_ndcg}\n")
outfile.write(f"map: {max_local_map}\n")
outfile.write('--------------\n\n')

for epoch, data in epoch_data.items():
    outfile.write(f"epoch: {epoch};\n")
    for k, value in data.items():
        outfile.write(f"Recall@{k}: {value['recall']}, NDCG@{k}: {value['ndcg']}, MAP@{k}: {value['map']}\n")
outfile.write("=============================")

outfile.close()
file.close()