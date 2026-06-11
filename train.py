import os
import argparse
import torch
import numpy as np
from CSCF import CSCF
from MultiViewDataset import MultiViewDataset, custom_collate_fn
from torch.utils.data import DataLoader

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='trained_models')
    parser.add_argument('--dataset', type=str, default='UCF', help='ActivityNet, Food101, SUNRGBD')
    parser.add_argument('--test_type', type=str, default='same', help='same, contra, data1, data2, total or all')
    parser.add_argument('--text_features', type=str, default='clipExtend', help='clip or clipExtend')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--topk', type=int, default=1)

    args = parser.parse_args()
    return args

def train(args):
    run_folder = os.getcwd()
    data_folder = os.path.join(run_folder, args.data_root)
    save_dir = os.path.join(run_folder, args.save_dir)
    seed = args.seed
    dataset = args.dataset
    epochs = args.epochs
    device = args.device
    text_features = args.text_features

    if dataset == 'ActivityNet':
        lr = 3e-3
    elif dataset == 'Food101':
        lr = 1e-3
    elif dataset == 'SUNRGBD':
        lr = 1e-4
    else:
        raise ValueError('Invalid dataset name')

    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'------------------ Loading train data ------------------')
    train_dataset = MultiViewDataset(data_folder=data_folder, dataset=dataset, text_features=text_features, type='same', mode='multi', train=True)
    attrs_multi = train_dataset.attrs_multi
    similarity_mat = train_dataset.similarity_mat
    modal_data = train_dataset.inputs_multi[:-2]
    train_dataloader = DataLoader(train_dataset, batch_size=2048, collate_fn=custom_collate_fn)

    print(f'------------------ Create model ------------------')
    num_class = attrs_multi.shape[0]
    attr_dim = attrs_multi.shape[1]
    hid_dim = 1024
    proto_dim = [datum.shape[1] for datum in modal_data]
    modal_num = len(modal_data)

    model = CSCF(attr_dim=attr_dim, hid_dim=hid_dim, proto_dim=proto_dim, modal_num=modal_num, lr=lr,
                 weight_decay=args.weight_decay, device=device, topk=args.topk, num_class=num_class)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")

    print(f'------------------ Start training ------------------')
    model.train_loop(train_dataloader, attrs_multi, similarity_mat, epochs=epochs)

    torch.save(model, os.path.join(save_dir, f'{args.dataset}_{args.text_features}.pth'))
    print(f'------------------ Finished training, saved to {save_dir} ------------------')

    return model

def test_step(args, model, data_folder, dataset, text_features, type, mode):
    test_dataset = MultiViewDataset(data_folder=data_folder, dataset=dataset, text_features=text_features, type=type, mode=mode, train=False)
    test_attrs_multi = test_dataset.attrs_multi
    test_similarity_mat = test_dataset.similarity_mat
    seen_masks = test_dataset.seen_masks
    test_dataloader = DataLoader(test_dataset, batch_size=2048, collate_fn=custom_collate_fn)

    if dataset == 'ActivityNet':
        balanced_weight = 5e-3
    elif dataset == 'Food101':
        balanced_weight = 3e-3
    elif dataset == 'SUNRGBD':
        balanced_weight = 1.5e-2
    else:
        raise ValueError('Invalid dataset name')

    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = [inputs for inputs in batch[0]]
            masks = [masks for masks in batch[2]]
            targets = torch.tensor(batch[1])

            probs = model(inputs, masks, test_attrs_multi, test_similarity_mat, seen_masks, balanced_weight)

            test_preds.extend(probs.argmax(dim=1).cpu().numpy())
            test_targets.extend(targets.cpu().numpy())

            del inputs, masks, targets, probs
            torch.cuda.empty_cache()

    test_acc = np.mean(np.array(test_preds) == np.array(test_targets))
    return test_acc


def test(args, model):
    run_folder = os.getcwd()
    data_folder = os.path.join(run_folder, args.data_root)
    seed = args.seed
    test_type = args.test_type
    dataset = args.dataset
    text_features = args.text_features
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'------------------ Test_type: {test_type} ------------------')
    if test_type in ['same', 'contra', 'data1', 'data2', 'total']:
        test_acc_audio = test_step(args, model, data_folder, dataset, text_features, test_type, mode='audio')
        test_acc_video = test_step(args, model, data_folder, dataset, text_features, test_type, mode='video')
        test_acc_multi = test_step(args, model, data_folder, dataset, text_features, test_type, mode='multi')
        if test_type == 'same':
            print(f'A_s acc: {test_acc_audio * 100: .2f}')
            print(f'B_s acc: {test_acc_video * 100: .2f}')
            print(f'A_s & B_s acc: {test_acc_multi * 100: .2f}')
        elif test_type == 'contra':
            print(f'A_u acc: {test_acc_audio * 100: .2f}')
            print(f'B_u acc: {test_acc_video * 100: .2f}')
            print(f'A_u & B_u acc: {test_acc_multi * 100: .2f}')
        elif test_type == 'data1':
            print(f'A_s acc: {test_acc_audio * 100: .2f}')
            print(f'B_u acc: {test_acc_video * 100: .2f}')
            print(f'A_s + B_u acc: {test_acc_multi * 100: .2f}')
        elif test_type == 'data2':
            print(f'A_u acc: {test_acc_audio * 100: .2f}')
            print(f'B_s acc: {test_acc_video * 100: .2f}')
            print(f'A_u + B_s acc: {test_acc_multi * 100: .2f}')
        else:
            print(f'A_all acc: {test_acc_audio * 100: .2f}')
            print(f'B_all acc: {test_acc_video * 100: .2f}')
            print(f'A_all + B_all acc: {test_acc_multi * 100: .2f}')
    elif test_type == 'mixture':
        test_acc_multi = test_step(args, model, data_folder, dataset, text_features, test_type, mode='multi')
        print(f'A_s & B_s & A_u & B_u & (A_all + B_all) acc: {test_acc_multi * 100: .2f}')
    elif test_type == 'all':
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='same', mode='multi')
        print(f'A_s & B_s acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='contra', mode='multi')
        print(f'A_u & B_u acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='data1', mode='audio')
        print(f'A_s acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='data1', mode='video')
        print(f'B_u acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='data1', mode='multi')
        print(f'A_s + B_u acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='data2', mode='audio')
        print(f'A_u acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='data2', mode='video')
        print(f'B_s acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='data2', mode='multi')
        print(f'A_u + B_s acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='total', mode='audio')
        print(f'A_all acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='total', mode='video')
        print(f'B_all acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='total', mode='multi')
        print(f'A_all + B_all acc: {test_acc * 100: .2f}')
        test_acc = test_step(args, model, data_folder, dataset, text_features, type='mixture', mode='multi')
        print(f'A_s & B_s & A_u & B_u & (A_all + B_all) acc: {test_acc * 100: .2f}')
    else:
        raise ValueError(f'test_type should be one of same, contra, data1, data2, total, mixture or all')


if __name__ == '__main__':
    args = get_parser()
    model = train(args)
    test(args, model)
