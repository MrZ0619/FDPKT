import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import argparse
import os
import datetime
from pathlib import Path

from model import FDPKT
from run import run_epoch

mp2path = {
    'BePKT': {
        'train_path': 'data/BePKT/train.csv',
        'valid_path': 'data/BePKT/valid.csv',
        'test_path': 'data/BePKT/test.csv',
        'train_skill_path': 'data/BePKT/train.csv',
        'valid_skill_path': 'data/BePKT/valid.csv',
        'test_skill_path': 'data/BePKT/test.csv',
        'skill_max': 101,
        'pro_max': 553,
        'err_feedback_max': 11
    },
    'Atcoder_C': {
        'train_path': 'data/Atcoder_C/train.csv',
        'valid_path': 'data/Atcoder_C/valid.csv',
        'test_path': 'data/Atcoder_C/test.csv',
        'train_skill_path': 'data/Atcoder_C/train.csv',
        'valid_skill_path': 'data/Atcoder_C/valid.csv',
        'test_skill_path': 'data/Atcoder_C/test.csv',
        'skill_max': 5277,
        'pro_max': 5277,
        'err_feedback_max': 9
    },
    'AIZU_Cpp': {
        'train_path': 'data/AIZU_Cpp/train.csv',
        'valid_path': 'data/AIZU_Cpp/valid.csv',
        'test_path': 'data/AIZU_Cpp/test.csv',
        'train_skill_path': 'data/AIZU_Cpp/train.csv',
        'valid_skill_path': 'data/AIZU_Cpp/valid.csv',
        'test_skill_path': 'data/AIZU_Cpp/test.csv',
        'skill_max': 9178,
        'pro_max': 9178,
        'err_feedback_max': 9
    }
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FDPKT Training Script')

    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix for save directory')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory for model files')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='Directory for result files')

    parser.add_argument('--use_response_enhancement', type=str2bool, default=True,
                        help='Enable response enhancement module')
    parser.add_argument('--use_response_change', type=str2bool, default=True,
                        help='Enable response change enhancement module')
    parser.add_argument('--use_diagnosis_router', type=str2bool, default=True,
                        help='Enable diagnosis router module')

    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--d', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=80,
                        help='Batch size')
    parser.add_argument('--min_seq', type=int, default=3,
                        help='Minimum sequence length')
    parser.add_argument('--max_seq', type=int, default=200,
                        help='Maximum sequence length')
    parser.add_argument('--grad_clip', type=float, default=15.0,
                        help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--cross_val_folds', type=int, default=5,
                        help='Number of cross-validation folds')

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.save_prefix:
        timestamp = f"{args.save_prefix}_{timestamp}"

    model_save_dir = Path(args.model_dir) / timestamp
    result_save_dir = Path(args.result_dir) / timestamp

    model_save_dir.mkdir(parents=True, exist_ok=True)
    result_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model save directory: {model_save_dir}")
    print(f"Result save directory: {result_save_dir}")

    config_path = result_save_dir / 'config.txt'
    with open(str(config_path), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FDPKT Training Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Directory Settings:\n")
        f.write(f"  save_prefix: {args.save_prefix}\n")
        f.write(f"  model_dir: {args.model_dir}\n")
        f.write(f"  result_dir: {args.result_dir}\n\n")
        f.write("Model Switches:\n")
        f.write(f"  use_response_enhancement: {args.use_response_enhancement}\n")
        f.write(f"  use_response_change: {args.use_response_change}\n")
        f.write(f"  use_diagnosis_router: {args.use_diagnosis_router}\n\n")
        f.write("Training Parameters:\n")
        f.write(f"  dropout: {args.dropout}\n")
        f.write(f"  d (embedding dimension): {args.d}\n")
        f.write(f"  learning_rate: {args.learning_rate}\n")
        f.write(f"  epochs: {args.epochs}\n")
        f.write(f"  batch_size: {args.batch_size}\n")
        f.write(f"  min_seq: {args.min_seq}\n")
        f.write(f"  max_seq: {args.max_seq}\n")
        f.write(f"  grad_clip: {args.grad_clip}\n")
        f.write(f"  patience: {args.patience}\n")
        f.write(f"  cross_val_folds: {args.cross_val_folds}\n")
        f.write("=" * 60 + "\n")

    for dataset in ['BePKT', 'Atcoder_C', 'AIZU_Cpp']:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_path = mp2path[dataset]['train_path']

        if 'valid_path' in mp2path[dataset]:
            valid_path = mp2path[dataset]['valid_path']
        else:
            valid_path = mp2path[dataset]['test_path']

        test_path = mp2path[dataset]['test_path']
        train_skill_path = mp2path[dataset]['train_skill_path']

        if 'valid_skill_path' in mp2path[dataset]:
            valid_skill_path = mp2path[dataset]['valid_skill_path']
        else:
            valid_skill_path = mp2path[dataset]['test_skill_path']

        test_skill_path = mp2path[dataset]['test_skill_path']
        skill_max = mp2path[dataset]['skill_max']
        err_feedback_max = mp2path[dataset]['err_feedback_max']

        if 'pro_max' in mp2path[dataset]:
            pro_max = mp2path[dataset]['pro_max']
        else:
            pro_max = skill_max

        print(pro_max, skill_max)
        p = args.dropout
        d = args.d
        learning_rate = args.learning_rate
        epochs = args.epochs
        batch_size = args.batch_size
        min_seq = args.min_seq
        max_seq = args.max_seq
        grad_clip = args.grad_clip
        patience = args.patience

        avg_auc = 0
        avg_acc = 0
        avg_rmse = 0

        sublist = []
        rmse_list = []

        for now_step in range(args.cross_val_folds):

            best_acc = 0
            best_auc = 0
            state = {'auc': 0, 'acc': 0, 'loss': 0}

            model = FDPKT(pro_max, skill_max, err_feedback_max, d, p,
                         use_response_enhancement=args.use_response_enhancement,
                         use_response_change=args.use_response_change,
                         use_diagnosis_router=args.use_diagnosis_router)
            model = model.to(device)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            one_p = 0

            for epoch in range(args.epochs):

                one_p += 1

                train_loss, train_acc, train_auc, train_rmse = run_epoch(pro_max, train_path, train_skill_path, batch_size,
                                                             True, min_seq, max_seq, model, optimizer, criterion,
                                                             device,
                                                             grad_clip)
                print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}, train_rmse: {train_rmse:.4f}')

                valid_loss, valid_acc, valid_auc, valid_rmse = run_epoch(pro_max, valid_path, valid_skill_path, batch_size, False,
                                                              min_seq, max_seq, model, optimizer, criterion, device,
                                                              grad_clip)

                print(f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_auc: {valid_auc:.4f}, valid_rmse: {valid_rmse:.4f}')

                sublist.append(valid_auc)
                rmse_list.append(valid_rmse)

                if valid_auc > best_auc:
                    one_p = 0
                    best_auc = valid_auc
                    best_acc = valid_acc
                    model_path = model_save_dir / f"FDPKT_{dataset}_{now_step}_model.pkl"
                    torch.save(model.state_dict(), str(model_path))
                    state['auc'] = valid_auc
                    state['acc'] = valid_acc
                    state['loss'] = valid_loss
                    state_path = model_save_dir / f'FDPKT_{dataset}_{now_step}_state.ckpt'
                    torch.save(state, str(state_path))

                if one_p >= patience:
                    break

            model_path = model_save_dir / f'FDPKT_{dataset}_{now_step}_model.pkl'
            model.load_state_dict(torch.load(str(model_path)))

            test_loss, test_acc, test_auc, test_rmse = run_epoch(pro_max, test_path, test_skill_path, batch_size, False,
                                                         min_seq, max_seq, model, optimizer, criterion, device,
                                                         grad_clip)

            print(f'*******************************************************************************')
            print(f'test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}, test_rmse: {test_rmse:.4f}')
            print(f'*******************************************************************************')

            avg_auc += test_auc
            avg_acc += test_acc
            avg_rmse += test_rmse

        avg_auc = avg_auc / args.cross_val_folds
        avg_acc = avg_acc / args.cross_val_folds
        avg_rmse = avg_rmse / args.cross_val_folds
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}, final_avg_rmse: {avg_rmse:.4f}')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')
        print(f'*******************************************************************************')

        output_path = result_save_dir / f'{dataset}_output.txt'
        with open(str(output_path), 'w') as file:
            file.write("AUC per epoch:\n")
            file.write('\n'.join(str(item) for item in sublist))
            file.write("\n\nRMSE per epoch:\n")
            file.write('\n'.join(str(item) for item in rmse_list))

        summary_path = result_save_dir / f'{dataset}_summary.txt'
        with open(str(summary_path), 'w') as f:
            f.write(f'dataset: {dataset}\n')
            f.write(f'final_avg_acc: {avg_acc:.4f}\n')
            f.write(f'final_avg_auc: {avg_auc:.4f}\n')
            f.write(f'final_avg_rmse: {avg_rmse:.4f}\n')
            f.write(f'timestamp: {timestamp}\n')
            f.write(f'save_prefix: {args.save_prefix}\n')
