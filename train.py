import argparse

from dataset import LabelMeDataset
from tool import mask2img, draw_results, cfg, Dict
from archt import BiSeNetV2, OhemCELoss, WarmupPolyLrScheduler

import torch
from torch.utils.tensorboard import SummaryWriter

cfg = Dict(**cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="/home/dominique/ownCloud/Shared/WRC_Videos/Results Dominique/DataSet")
    parser.add_argument('--out', type=str, default="")
    parser.add_argument('--path', type=str, default="model2.pth")
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--augmentation', action='store_true', default=False)
    return parser.parse_args()


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


if __name__ == '__main__':
    args = parse_args()
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # input (B*C*H*W)

    # dataset
    dataset = LabelMeDataset(args.dataset, args.augmentation, 544, 960)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    model = BiSeNetV2(n_classes=dataset.un_class)  # model.init_weights()
    model.to(device)
    # Set Model: set no parallel, cuda, train mode
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # not needed
    # model.cuda()    # device()
    model.train()
    # loss_fn = LossWithAux(nn.BCEWithLogitsLoss())
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]

    # construct an optimizer and a learning rate scheduler
    optimizer = set_optimizer(model)

    # meters
    # time_meter, loss_meter, loss_pre_meter, loss_aux_meters = get_meters()

    # lr scheduler
    lr_scheduler = WarmupPolyLrScheduler(optimizer, max_iter=cfg.max_iter,
                                         power=0.9, warmup_iter=cfg.warmup_iters,
                                         warmup_ratio=0.1, warmup='exp', last_epoch=-1, )

    print(f"Start training with {len(dataset)} images ...")
    for epoch in range(args.epochs):
        # train loop
        for it, (im, lb) in enumerate(trainloader):
            # try:
            im, lb = im.to(device), lb.to(device)
            # lb:: BxCxHxW vs BxHxW
            # im = im.cuda()
            # lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optimizer.zero_grad()
            logits, *logits_aux = model(im)
            loss_pre = criteria_pre(logits, lb)
            loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)  # loss_pre + 0.5*sum(loss_aux)
            loss.backward()
            optimizer.step()
            # torch.cuda.synchronize()
            lr_scheduler.step()

            # Metrics: time, loss, ...

            # print training log message
            lr = lr_scheduler.get_lr()

            print(f"Train at iteration {it} with lr: {sum(lr) / len(lr)}, loss: {loss.item()} and loss prediction: {loss_pre.item()}")
            # except (ValueError, RuntimeError, TypeError, NameError):
            #     print(f"Error at index: {idx}")
            #     print(f"Error at {dataset.getpath(idx=idx[0])} or ")
            #     print(f"Error at {dataset.getpath(idx=idx[1])}")

        torch.save(model.state_dict(), args.path)

    # ## dump the final model and evaluate the result
    #
    # # get some random training images
    # images, labels = next(iter(trainloader))
    # img_grid = torchvision.utils.make_grid(images)
    # lbl_grid = torchvision.utils.make_grid(labels)
    #
    # # default `log_dir` is "runs" - we'll be more specific here
    # writer_img = SummaryWriter('runs/img')
    # writer_lbl = SummaryWriter('runs/lbl')
    # writer = SummaryWriter('runs/wrc')
    #
    # # write to tensorboard
    # writer_img.add_image('img', img_grid)
    # writer_lbl.add_image('lbl', lbl_grid)
    # writer.add_graph(model, images)

    print("Done")
