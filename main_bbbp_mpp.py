import os
# os.environ['CUDA_VISIBLE_DEVICES'] = f"0"
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_adj
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn import GNN
from gru import SeqModel


from tqdm import tqdm
import argparse
import time
import numpy as np
import random
from datetime import datetime
from ogb.graphproppred import Evaluator
now = datetime.now()
timestamp = str(now.year)[-2:] + "_" + str(now.month).zfill(2) + "_" + str(now.day).zfill(2) + "_" + \
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

### importing OGB-LSC
cls_criterion = torch.nn.BCEWithLogitsLoss()

from torch_geometric.utils import dropout_adj

def contrastive_loss(z1, z2, temperature):
    # z1과 z2는 같은 그래프의 다른 증강 뷰에 대한 임베딩입니다.
    # 이들 간의 코사인 유사도를 계산합니다.
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    similarity_matrix = torch.mm(z1, z2.T) / temperature
    # 대각선(자기 자신과의 비교)에 대한 손실을 계산합니다.
    loss = -torch.log(torch.exp(similarity_matrix.diag()) / torch.exp(similarity_matrix).sum(dim=1))
    return loss.mean()

def custom_dropout_edges(edge_index, edge_attr, drop_prob, num_nodes):
    # 무작위 드롭 마스크 생성
    drop_mask = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_prob
    
    # 드롭 마스크를 사용하여 엣지 인덱스와 속성을 필터링
    edge_index = edge_index[:, drop_mask]
    edge_attr = edge_attr[drop_mask]

    return edge_index, edge_attr

def augment_data(batch, device, node_drop_prob=0.1, edge_drop_prob=0.1):
    # 모든 텐서를 지정된 디바이스로 이동
    batch.to(device)

    # 노드 드롭: 무작위로 노드의 일부 특성을 0으로 만듭니다.
    node_drop_mask = torch.rand(batch.x.size(0), device=device) > node_drop_prob
    batch.x = batch.x * node_drop_mask.unsqueeze(1)

    # 엣지 드롭: 무작위로 그래프의 일부 엣지를 제거합니다.
    edge_index, edge_attr = custom_dropout_edges(batch.edge_index, batch.edge_attr, edge_drop_prob, batch.num_nodes)
    # edge_index, edge_mask = dropout_adj(batch.edge_index, p=edge_drop_prob, num_nodes=batch.num_nodes)
    # # 엣지 속성 업데이트: 드롭되지 않은 엣지에 해당하는 엣지 속성만 유지합니다.
    # batch.edge_attr = batch.edge_attr[edge_mask]

    # 특성 마스킹: 무작위로 노드의 일부 특성을 마스킹합니다.
    feature_mask = torch.rand(batch.x.size(), device=device) > edge_drop_prob
    batch.x = batch.x * feature_mask

    # 드롭된 엣지 인덱스와 속성으로 배치 업데이트
    augmented_batch = batch
    augmented_batch.edge_index = edge_index.to(device)
    augmented_batch.edge_attr = edge_attr.to(device)

    return augmented_batch

def augment_data_node(batch, device, node_drop_prob=0.1):
    # 모든 텐서를 지정된 디바이스로 이동
    batch.to(device)

    # 노드 드롭: 무작위로 노드의 일부 특성을 0으로 만듭니다.
    node_drop_mask = torch.rand(batch.x.size(0), device=device) > node_drop_prob
    batch.x = batch.x * node_drop_mask.unsqueeze(1)

    # 특성 마스킹: 무작위로 노드의 일부 특성을 마스킹합니다.
    feature_mask = torch.rand(batch.x.size(), device=device) > node_drop_prob
    batch.x = batch.x * feature_mask

    return batch

def augment_data_edge(batch, device, edge_drop_prob=0.1):
    # 모든 텐서를 지정된 디바이스로 이동
    batch.to(device)

    # 엣지 드롭: 무작위로 그래프의 일부 엣지를 제거합니다.
    edge_index, edge_attr = custom_dropout_edges(batch.edge_index, batch.edge_attr, edge_drop_prob, batch.num_nodes)
    
    # 드롭된 엣지 인덱스와 속성으로 배치 업데이트
    batch.edge_index = edge_index.to(device)
    batch.edge_attr = edge_attr.to(device)

    return batch

def train(model, device, loader, optimizer, evaluator):
    model.train()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:

            is_labeled = batch.y == batch.y
            
            pred = model(batch)
            # 두 가지 다른 증강된 데이터 생성
            augmented_batch_node = augment_data_node(batch, device, node_drop_prob=0.1)
            augmented_batch_edge = augment_data_edge(batch, device, edge_drop_prob=0.1)

            pred1 = model(augmented_batch_node)
            pred2 = model(augmented_batch_edge)

            # 일반적인 지도 학습 손실 및 대조 손실 계산
            supervised_loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            unsupervised_loss = contrastive_loss(pred1, pred2, temperature=0.1)
        
            # 혼합 손실을 계산하여 역전파
            loss = supervised_loss + unsupervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_true.append(batch.y.view(pred1.shape).detach().cpu())
            y_pred.append(pred1.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--gru_emb', type=int, default=32,
                        help='GRU token embed size (default: 32)')
    parser.add_argument('--gru_hid', type=int, default=64,
                        help='GRU hidden size (default: 256)')
    parser.add_argument('--max_len', type=int, default=500,
                        help='')
    parser.add_argument('--train_subset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='num_labels, tox21: 12, PCQM4: 1')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    #parser.add_argument('--checkpoint_dir', type=str, default=f'ckpt/{timestamp}', help='directory to save checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt/finetune_{}'.format(timestamp), help='directory to save checkpoint')

    args = parser.parse_args()

    print(args)



    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    train_dataset = torch.load("dataset/bbbp/processed/train_dataset.pt")
    valid_dataset = torch.load("dataset/bbbp/processed/valid_dataset.pt")
    test_dataset = torch.load("dataset/bbbp/processed/test_dataset.pt")

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator("ogbg-molbbbp")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        "num_tasks": args.num_tasks,
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gru':
        model = SeqModel(args).to(device)
    else:
        raise ValueError('Invalid GNN type')

    # 프리트레인된 모델의 상태 사전 로드
    pretrained_model_path = './ckpt/23_12_04_120829/checkpoint.pt'  # 실제 파일 경로로 수정
    checkpoint = torch.load(pretrained_model_path, map_location=device)

    # 모델의 가중치만 포함하는 상태 사전을 추출
    pretrained_state_dict = checkpoint['model_state_dict']

    # 상태 사전을 현재 모델에 적용
    model.load_state_dict(pretrained_state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    #print(f'#Params: {num_params}')
    print('#Params: {}'.format(num_params))


    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_auc = 0
    best_epoch = 0

    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_auc = train(model, device, train_loader, optimizer, evaluator)
        train_auc = train_auc["rocauc"]
        print('Evaluating...')
        valid_auc = eval(model, device, valid_loader, evaluator)
        valid_auc = valid_auc["rocauc"]
        print({'Train': train_auc, 'Validation': valid_auc})

        if args.log_dir != '':
            writer.add_scalar('valid/auc', valid_auc, epoch)
            writer.add_scalar('train/auc', train_auc, epoch)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_auc': best_valid_auc,
                              'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            test_auc = eval(model, device, test_loader, evaluator)
            #print(f"Test MAE: {test_auc}")
            print("Test MAE: {}".format(test_auc))

        scheduler.step()

        #print(f'Best validation MAE so far: Epoch {best_epoch}: {best_valid_auc}')
        print('Best validation MAE so far: Epoch {}: {}'.format(best_epoch, best_valid_auc))

    #print(f"Test AUC: {test_auc}")
    print("Test AUC: {}".format(test_auc))
    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
