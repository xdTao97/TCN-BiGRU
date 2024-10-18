import json
from datetime import datetime

import torch
import torch.nn as nn
import time
from args import get_parser

# from model import TCNSENetBiGRUGlobalAttention

from model_Ablation import TCNSENetBiGRUGlobalAttention
from pltFeature import featureimportance_all_samples
from utils import *
from prediction2 import Predictor
from train2 import Trainer

if __name__ == "__main__":

    # time
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    start = time.time()
    # parameter
    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(f"arg_summary: {args_summary}")


    def load_dataset(dataFile):
        # folder = os.path.join(dataFile, dataset)
        folder = dataFile
        print(folder)
        if not os.path.exists(folder):
            raise Exception('Processed Data not found.')
        loader = []
        for file in ['train', 'test', 'labels']:
            loader.append(np.load(os.path.join(folder, f'{file}.npy')))
        # loader = [i[:, debug:debug+1] for i in loader]
        if args.less: loader[0] = cut_array(0.2, loader[0])
        # train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
        # test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
        labels = loader[2]

        return loader[0], loader[1], labels


    #filenameStart = "output_windowSize_SNR"
    #filenameStart = "output_windowSize"
    filenameStart = "output_Ablation"
    if dataset in ['MSL', 'SMAP']:
        print("1")
        output_path = f'{filenameStart}/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
        # print(f"x_train: {x_train}")
    elif dataset == 'SWAT':
        output_path = f'{filenameStart}/{dataset}'
        dataFile = 'datasets/SWaT/processed'
        # train_loader, test_loader, labels = load_dataset(args.dataset)
        x_train, x_test, y_test = load_dataset(dataFile)
    elif dataset == 'SMD':
        output_path = f'{filenameStart}/SMD'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # log path
    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    # x_train  convert to  tensor
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]
    print(f"n_features: {n_features}")

    target_dims = get_target_dims(dataset)
    print(f"target_dims: {target_dims}")
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    # Will forecast and reconstruct input feature dim
    print(f"out_dim: {out_dim}")

    # 定义模型参数
    batch_size = 256
    input_dim = n_features  # 输入的特征维度
    output_dim = out_dim  # 输出的特征维度
    num_channels = [32, 64]  # 每个TemporalBlock中的输出通道数
    kernel_size = 4  # 卷积核大小
    dropout = 0.2  # Dropout概率
    # BiGRU 层数和维度数
    # hidden_layer_sizes = [32, 64]
    hidden_layer_sizes = [32, 64]
    # 全局注意力维度数
    attention_dim = hidden_layer_sizes[-1]  # 注意力层维度 默认为 BiGRU输出层维度

    # Dataloader
    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("args.seed:", args.seed)

    # model
    model = TCNSENetBiGRUGlobalAttention(window_size, batch_size, input_dim, output_dim,
                                         num_channels, kernel_size, hidden_layer_sizes,
                                         attention_dim, dropout, gru_hid_dim=150, recon_hid_dim=150, recon_n_layers=1)
    model = model.cuda()

    # print(model)
    # model = model.cuda()  
    # summary(model, (100, 25))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()
    print("--------------------------------------------")
    print("--------------------------------------------")
    # recon_criterion = nn.MSELoss()
    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    # print(trainer.model)
    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)


    #test_loss = trainer.evaluate_SNR(test_loader)
    test_loss = trainer.evaluate(test_loader)
    print(test_loss)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test total loss: {test_loss[1]:.5f}")
    # #
    # # # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "SWAT": (0.993, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q
    #
    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1, "SWAT": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]
    # 加在训练好的模型
    # save_path = "output/SMAP/04012022_194215"
    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
        args_summary
    )
    #
    label = y_test[window_size:] if y_test is not None else None
    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    end = time.time()
    m, s = divmod(end - start, 60)
    all_time = {'runtime': "%02d:%02d" % (m, s)}
    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, indent=2)
        json.dump(all_time, f, indent=2, ensure_ascii=False)

    #featureimportance_all_samples(model, train_loader, input_dim)
