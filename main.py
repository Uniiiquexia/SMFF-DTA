import random,os
from datetime import datetime
from dataset import CustomDataSet, collate_fn
from model import SMFFDTA
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import mean_squared_error,mean_absolute_error
from emetrics import get_rm2, get_cindex


def test_precess(model,pbar):
    loss_f = nn.MSELoss()
    model.eval()
    test_losses = []
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, fps, atoms, proteins, sse, phyche, labels = data
            compounds = compounds.cuda()
            fps = fps.cuda()
            atoms = atoms.cuda()
            proteins = proteins.cuda()
            sse = sse.cuda()
            phyche = phyche.cuda()
            labels = labels.cuda()
            predicts= model.forward(compounds, fps, atoms, proteins, sse, phyche)
            loss = loss_f(predicts, labels.view(-1, 1))
            total_preds = torch.cat((total_preds, predicts.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)
            test_losses.append(loss.item())
    Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
    test_loss = np.average(test_losses)
    return Y, P, test_loss, mean_squared_error(Y, P), mean_absolute_error(Y, P), get_rm2(Y, P), get_cindex(Y, P)

def test_model(test_dataset_load,save_path,DATASET,lable = "Train",save = True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(test_dataset_load)),
        total=len(test_dataset_load))
    T, P, loss_test, mse_test, mae_test, rm2_test, c_index_test= test_precess(model,test_pbar)
    if save:
        with open(save_path + "/{}_stable_{}_prediction.txt".format(DATASET,lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};MSE:{:.5f};MAE:{:.5f};Rm2:{:.5f};c_index:{:.5f}.' \
        .format(lable, loss_test, mse_test, mae_test, rm2_test, c_index_test)
    print(results)
    return results,mse_test, mae_test, rm2_test, c_index_test

def get_kfold_data(i, datasets, k=5):

    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:] 
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0")

if __name__ == "__main__":
    
    start = datetime.now()
    print('start at ', start)
    
    """select seed"""
    SEED = 4321  
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """Load preprocessed data."""
    DATASET = "Davis"
    # DATASET = "KIBA"
    print("Train in {}".format(DATASET))
    tst_path = f'../SMFF-DTA/Dataset/{DATASET}/{DATASET}.txt'
    
    with open(tst_path, 'r') as f:
        cpi_list = f.read().strip().split('\n')
    print("load finished")
    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(cpi_list, SEED)
    # random.shuffle(cpi_list)
    K_Fold = 5
    Batch_size = 32 
    weight_decay = 1e-4 
    Learning_rate = 5e-6
    Patience = 20    
    Epoch = 300     
    """Output files."""
    save_path = "../SMFF-DTA/Results/{}/".format(DATASET)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + 'The_results.txt'

    MSE_List, MAE_List, Rm2_List, c_index_List = [], [], [], []
    
    """ create model"""
    model = SMFFDTA()
    model = model.to(device)
    # print(model)
    model_path = Path(f'../SMFF-DTA/saveModel/SMFFDTA_{datetime.now().strftime("%Y%m%d%H%M%S")}')
    if not os.path.exists(model_path):
        os.makedirs(model_path)    
    f_model = open(model_path / "model.txt", 'w')
    f_model.write('model: \n')
    f_model.write(str(model)+'\n')
    f_model.close()

    for i_fold in range(K_Fold):
        print('*' * 25, 'fold', i_fold + 1, '*' * 25)
        trainset, testset = get_kfold_data(i_fold, dataset, k=K_Fold)
        TVdataset = CustomDataSet(trainset)
        test_dataset = CustomDataSet(testset)
        TVdataset_len = len(TVdataset)
        valid_size = int(0.2 * TVdataset_len)
        train_size = TVdataset_len - valid_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])
        train_dataset_load = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=2,
                                        collate_fn=collate_fn)
        valid_dataset_load = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False, num_workers=2,
                                        collate_fn=collate_fn)
        test_dataset_load = DataLoader(test_dataset, batch_size=Batch_size, shuffle=False, num_workers=2,
                                       collate_fn=collate_fn)

        """weight initialize"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        LOSS_F = nn.MSELoss()
        
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=Learning_rate)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=Learning_rate, max_lr=Learning_rate * 10,
                                                cycle_momentum=False,
                                                step_size_up=train_size // Batch_size)
        
        save_path_i = "{}/{}_Fold/".format(save_path, i_fold + 1)
        if not os.path.exists(save_path_i):
            os.makedirs(save_path_i)
        note = ""
        writer = SummaryWriter(log_dir=save_path_i, comment=note)

        """Start training."""
        print('Training...')
        start = timeit.default_timer()
        patience = 0
        best_score = 100
        for epoch in range(1, Epoch + 1):
            train_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(train_dataset_load)),
                total=len(train_dataset_load))
            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                '''data preparation '''
                train_compounds, train_fps, train_atoms, train_proteins, train_sse, train_phyche, train_labels = train_data
                train_compounds = train_compounds.cuda()
                train_fps = train_fps.cuda()
                train_atoms = train_atoms.cuda()
                train_proteins = train_proteins.cuda()
                train_sse = train_sse.cuda()
                train_phyche = train_phyche.cuda()
                train_labels = train_labels.cuda()

                optimizer.zero_grad()

                predicts = model.forward(train_compounds, train_fps, train_atoms, train_proteins, train_sse, train_phyche)
                train_loss = LOSS_F(predicts, train_labels.view(-1, 1))
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()

                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(train_losses_in_epoch)
            writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)

            """valid"""
            valid_pbar = tqdm(
                enumerate(
                    BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))
            valid_losses_in_epoch = []
            model.eval()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:
                    '''data preparation '''
                    valid_compounds, valid_fps, valid_atoms, valid_proteins, valid_sse, valid_phyche, valid_labels = valid_data

                    valid_compounds = valid_compounds.cuda()
                    valid_fps = valid_fps.cuda()
                    valid_atoms = valid_atoms.cuda()
                    valid_proteins = valid_proteins.cuda()
                    valid_sse = valid_sse.cuda()
                    valid_phyche = valid_phyche.cuda()
                    valid_labels = valid_labels.cuda()
                    valid_predictions = model.forward(valid_compounds, valid_fps, valid_atoms, valid_proteins, valid_sse, valid_phyche)
                    valid_loss = LOSS_F(valid_predictions, valid_labels.view(-1, 1))
                    valid_losses_in_epoch.append(valid_loss.item())
                    total_preds = torch.cat((total_preds, valid_predictions.cpu()), 0)
                    total_labels = torch.cat((total_labels, valid_labels.cpu()), 0)
                Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
            valid_MSE = mean_squared_error(Y, P)
            valid_MAE = mean_absolute_error(Y, P)
            valid_Rm2 = get_rm2(Y, P)
            valid_c_index = get_cindex(Y, P)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)  

            if valid_MSE < best_score:
                best_score = valid_MSE
                patience = 0
                torch.save(model.state_dict(), save_path_i + 'valid_best_checkpoint.pth')
            else:
                patience+=1
            epoch_len = len(str(Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_MSE: {valid_MSE:.5f} ' +
                         f'valid_MAE: {valid_MAE:.5f} ' +
                         f'valid_Rm2: {valid_Rm2:.5f} ' )
            print(print_msg)
            writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
            writer.add_scalar('Valid MSE', valid_MSE, epoch)
            writer.add_scalar('Valid MAE', valid_MAE, epoch)
            writer.add_scalar('Valid Rm2', valid_Rm2, epoch)
            writer.add_scalar('Valid c_index', valid_c_index, epoch)

            if patience == Patience:
                break
        torch.save(model.state_dict(), save_path_i + 'stable_checkpoint.pth')
        """load trained model"""
        model.load_state_dict(torch.load(save_path_i + "valid_best_checkpoint.pth"))
        trainset_test_results,_,_,_,_ = test_model(train_dataset_load, save_path_i, DATASET, lable="Train")
        validset_test_results,_,_,_,_ = test_model(valid_dataset_load, save_path_i, DATASET, lable="Valid")
        testset_test_results,mse_test, mae_test, rm2_test, c_index_test = test_model(test_dataset_load,save_path_i,DATASET,lable="Test")
        
        with open(save_path + "The_results.txt", 'a') as f:
            f.write("results on {}th fold\n".format(i_fold+1))
            f.write(trainset_test_results + '\n')
            f.write(validset_test_results + '\n')
            f.write(testset_test_results + '\n')
        writer.close()
        MSE_List.append(mse_test)
        MAE_List.append(mae_test)
        Rm2_List.append(rm2_test)
        c_index_List.append(c_index_test)
    MSE_mean, MSE_var = np.mean(MSE_List), np.sqrt(np.var(MSE_List))
    MAE_mean, MAE_var = np.mean(MAE_List), np.sqrt(np.var(MAE_List))
    Rm2_mean, Rm2_var = np.mean(Rm2_List), np.sqrt(np.var(Rm2_List))
    c_index_mean, c_index_var = np.mean(c_index_List), np.sqrt(np.var(c_index_List))
    with open(save_path + 'The_results.txt', 'a') as f:
        f.write('The results on {}:'.format(DATASET) + '\n')
        f.write('MSE(std):{:.4f}({:.4f})'.format(MSE_mean, MSE_var) + '\n')
        f.write('MAE(std):{:.4f}({:.4f})'.format(MAE_mean, MAE_var) + '\n')
        f.write('Rm2(std):{:.4f}({:.4f})'.format(Rm2_mean, Rm2_var) + '\n')
        f.write('c_index(std):{:.4f}({:.4f})'.format(c_index_mean, c_index_var) + '\n')
    print('MSE(std):{:.4f}({:.4f})'.format(MSE_mean, MSE_var))
    print('MAE(std):{:.4f}({:.4f})'.format(MAE_mean, MAE_var))
    print('Rm2(std):{:.4f}({:.4f})'.format(Rm2_mean, Rm2_var))
    print('c_index(std):{:.4f}({:.4f})'.format(c_index_mean, c_index_var))
    

end = datetime.now()
print('end at:', end)
