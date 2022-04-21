import numpy as np

import torch
import pickle
import time
import os

from TFPRec import TFPRec


def train_TFPRec(train_set, test_set, h_params, vocab_size, device, city_code):

    model_path = f"./results/{city_code}_model"
    log_path = f"./results/{city_code}_log"
    meta_path = f"./results/{city_code}_meta"

    print("parameters:", h_params)
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()

    # construct model
    rec_model = TFPRec(
        vocab_size = vocab_size,
        f_embed_size = h_params['embed_size'],
        num_encoder_layers = h_params['tfp_layer_num'],
        num_lstm_layers = h_params['lstm_layer_num'],
        num_heads = h_params['head_num'],
        forward_expansion = h_params['expansion'],
        dropout_p = h_params['dropout'],
        back_step = h_params['future_step'],
        aux_train = h_params['aux'],
        mask_prop = h_params['mask_prop']
    )

    rec_model = rec_model.to(device)

    # continue with previous training
    start_epoch = 0
    if os.path.isfile(model_path):
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()

        # load training epoch
        meta_file = open(meta_path, "rb")
        start_epoch=pickle.load(meta_file) + 1
        meta_file.close()

    params = list(rec_model.parameters())
    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}

    for i in range(start_epoch, h_params['epoch']):
        begin_time = time.time()
        total_loss = 0.
        for sample in train_set:
            
            sample_to_device = []
            for seq in sample:
                features = torch.tensor(seq[:5]).to(device)
                dist_matrix = torch.tensor(seq[5]).to(device)
                
                sample_to_device.append((features, dist_matrix))
            
            loss, _ = rec_model(sample_to_device)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test
        # if i%10==0:
        recall, ndcg, map = test_TFPRec(test_set, rec_model)
        recalls[i] = recall
        ndcgs[i] = ndcg
        maps[i] = map
        
        # record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[i] = avg_loss
        print(f"epoch: {i}; average loss: {avg_loss}, time taken: {int(time.time()-begin_time)}s")
        # save model
        torch.save(rec_model.state_dict(), model_path)
        # save last epoch
        meta_file = open(meta_path, 'wb')
        pickle.dump(i, meta_file)
        meta_file.close()
        
        
        # early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss)>10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {i}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        file.close()

    print("============================")


def test_TFPRec(test_set, rec_model, ks=[1,5,10]):

    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels==preds[:,:k], dim=1))/labels.shape[0]
    
    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:,:k] == labels).nonzero()[:,1] + 1
        ndcg = 1/torch.log2(exist_pos+1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:,:k] == labels).nonzero()[:,1] + 1
        map = 1/exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    for sample in test_set:
        sample_to_device = []
        for seq in sample:
            features = torch.tensor(seq[:5]).to(device)
            dist_matrix = torch.tensor(seq[5]).to(device)
            
            sample_to_device.append((features, dist_matrix))
        
        pred, label = rec_model.predict(sample_to_device)
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")
    
    return recalls, NDCGs, MAPs


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    city_index = input("Please select the city: \n\n1. New York, \n2. Singapore, \n3. Pheonix\n")
    if city_index == '1':
        city = 'NY'
    elif city_index == '2':
        city = 'SIN'
    elif city_index == '3':
        city = 'PHO'
    else:
        raise Exception("Invalid City Code Selected")

    # get parameters
    h_params = {}
    h_params['expansion'] = 4
    h_params['future_step'] = 1
    h_params['aux'] = True
    h_params['mask_prop'] = 0.1
    h_params['lr'] = 1e-4
    h_params['epoch'] = 100
    h_params['loss_delta'] = 1e-3


    # read training data
    file = open(f"./processed_data/{city}_train", 'rb')
    train_set=pickle.load(file)
    file = open(f"./processed_data/{city}_valid", 'rb')
    valid_set=pickle.load(file)

    # read meta data
    file = open(f"./processed_data/{city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {}
    vocab_size["POI"] = torch.tensor(len(meta["POI"])).to(device)
    vocab_size["cat"] = torch.tensor(len(meta["cat"])).to(device)
    vocab_size["user"] = torch.tensor(len(meta["user"])).to(device)
    vocab_size["hour"] = torch.tensor(len(meta["hour"])).to(device)
    vocab_size["day"] = torch.tensor(len(meta["day"])).to(device)
    
    # adjust specific parameters for each city
    if city == 'SIN':
        # SIN param
        h_params['embed_size'] = 20
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 3
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1

    elif city == 'PHO':
        # PHO param
        h_params['embed_size'] = 20
        h_params['tfp_layer_num'] = 4
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1

    elif city == 'NY':
        # NY param
        h_params['embed_size'] = 20
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.1
        h_params['head_num'] = 1

    # create output folder
    if not os.path.isdir('./results'):
        os.mkdir("./results") 

    train_TFPRec(train_set, valid_set, h_params, vocab_size, device, city_code=city)