import copy
import os
import sys
from datetime import datetime
from itertools import cycle

import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import StepLR
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from loss import LossComputer

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from divdis import DivDisLoss

device = torch.device("cuda")


def sec_to_str(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    h, m, s = int(h), int(m), int(s)
    return f"{h:02}:{m:02}:{s:02}"


def run_epoch_divdis_eval(
    epoch, model, loader, loss_computers, logger, csv_logger, args
):
    model.eval()

    ys, yhats, gs = [], [], []
    for batch_idx, batch in enumerate(loader):
        if args.debug and batch_idx > 3:
            break

        x, y, g, _ = batch
        y_cp, g_cp = copy.deepcopy(y), copy.deepcopy(g)
        del y, g
        y, g = y_cp, g_cp
        x, y, g = x.cuda(), y.cuda(), g.cuda()
        y_alt = (g % 2).cuda()

        yhat = model(x)
        yhat_chunked = torch.chunk(yhat, args.heads, dim=-1)

        for i, _yhat in enumerate(yhat_chunked):
            loss_computers[f"h{i}"].loss(_yhat, y, g, False, y_onehot=None)
            loss_computers[f"h{i}_alt"].loss(_yhat, y_alt, g, False, y_onehot=None)

        ys.append(y.cpu())
        yhats.append(yhat.cpu())
        gs.append(g.cpu())

    all_stats = {}
    for computer_idx, loss_computer in loss_computers.items():
        stats = loss_computer.get_stats()
        all_stats.update({f"{computer_idx}_{k}": v for k, v in stats.items()})
        loss_computer.reset_stats()
    csv_logger.log(epoch, batch_idx, all_stats)
    csv_logger.flush()

    worst_keys = [k for k in all_stats.keys() if "worst" in k and "alt" not in k]
    worst_vals = [all_stats[k] for k in worst_keys]
    avg_keys = [k for k in all_stats.keys() if "group_avg_acc" in k and "alt" not in k]
    avg_vals = [all_stats[k] for k in avg_keys]
    delta = (datetime.now() - args.start_time).total_seconds()
    N = len(loader) * args.n_epochs
    n = epoch * len(loader) + batch_idx
    delta_est = delta * N / n
    logger.write(
        f"Elapsed: {sec_to_str(delta)}/{sec_to_str(delta_est)} Epoch {epoch} batch {batch_idx+1}/{len(loader)}\t"
    )
    logger.write(
        f"Avg acc {max(avg_vals)*100:.1f}, Worst acc {max(worst_vals)*100:.1f}\n"
    )
    logger.flush()

    all_ys = torch.cat(ys)
    all_gs = torch.cat(gs)
    all_yhats = torch.chunk(torch.cat(yhats), args.heads, dim=-1)

    probs_stacked = torch.stack(all_yhats).softmax(-1)
    diffs = probs_stacked.unsqueeze(0) - probs_stacked.unsqueeze(1)
    disagreement_order = diffs.abs().sum([0, 1, 3]).argsort(descending=True)

    def get_avg_and_worst_accs(query_idxs):
        y_i = all_ys[query_idxs]
        accs = [
            (yhat[query_idxs].argmax(dim=1) == y_i).float().mean() for yhat in all_yhats
        ]
        best_idx = np.argmax(accs)
        return avg_vals[best_idx], worst_vals[best_idx]

    for N in [4, 8, 16, 32, 64, 128]:
        # Active querying
        active_idxs = disagreement_order[:N]
        act_avg_acc, act_worst_acc = get_avg_and_worst_accs(active_idxs)
        logger.write(f"{N=} active query {act_avg_acc} {act_worst_acc}\n")

        # Random querying
        avg_results, worst_results = [], []
        for _ in range(100):
            rand_idxs = np.random.choice(all_ys.shape[0], N)
            avg_acc, worst_acc = get_avg_and_worst_accs(rand_idxs)
            avg_results.append(avg_acc)
            worst_results.append(worst_acc)
        logger.write(f"{N=} random query {np.mean(avg_results)} {np.mean(worst_results)}\n")


def run_epoch_divdis_train(
    epoch,
    model,
    optimizer,
    loader,
    loader_unlabeled,
    loss_computers,
    logger,
    csv_logger,
    args,
    is_training,
    log_every=50,
    scheduler=None,
):
    assert is_training
    loss_fn = DivDisLoss(heads=args.heads, mode=args.mode, reduction=args.reduction)

    model.train()
    if "bert" in args.model:
        model.zero_grad()

    if args.fixed_label_ratio:
        unlabeled_zero_freq = args.fixed_label_ratio
    else:
        if args.ratio_split == "source":
            ratio_dataset = loader.dataset
        elif args.ratio_split == "target":
            ratio_dataset = loader_unlabeled.dataset
        unlabeled_labels = ratio_dataset._y_array.int()
        counts = torch.bincount(unlabeled_labels)
        unlabeled_zero_freq = counts[0] / counts.sum()
    print(f"Label ratio for regularizer: {unlabeled_zero_freq}")

    loader_both = zip(loader, cycle(loader_unlabeled))
    for batch_idx, (batch, batch_unlabeled) in enumerate(loader_both):
        if args.debug and batch_idx > 3:
            break

        x, y, g, _ = batch
        y_cp, g_cp = copy.deepcopy(y), copy.deepcopy(g)
        del y, g
        y, g = y_cp, g_cp
        x, y, g = x.cuda(), y.cuda(), g.cuda()
        y_alt = (g % 2).cuda()
        x_unlabeled, *_ = batch_unlabeled
        x_unlabeled = x_unlabeled.cuda()

        if args.bn_mode == "train":
            yhat = model(x)
            yhat_unlabeled = model(x_unlabeled)
        elif args.bn_mode == "eval":
            model.train()
            yhat = model(x)
            model.eval()
            yhat_unlabeled = model(x_unlabeled)
        elif args.bn_mode == "mix":
            bs_x = x.shape[0]
            x_both = torch.cat([x, x_unlabeled], dim=0)
            y_both = model(x_both)
            yhat, yhat_unlabeled = y_both[:bs_x], y_both[bs_x:]
        else:
            raise ValueError("Batchnorm mode {args.bn_mode} not implemented!")

        all_loss_mains = []
        yhat_chunked = torch.chunk(yhat, args.heads, dim=-1)
        for i, _yhat in enumerate(yhat_chunked):
            loss_main = loss_computers[f"h{i}"].loss(
                _yhat, y, g, is_training, y_onehot=None
            )
            all_loss_mains.append(loss_main)

            loss_computers[f"h{i}_alt"].loss(
                _yhat, y_alt, g, is_training, y_onehot=None
            )
        loss_main = torch.stack(all_loss_mains).mean()

        repulsion_loss = loss_fn(yhat_unlabeled)
        loss_main += repulsion_loss * args.diversity_weight

        yhat_unlabeled_chunked = torch.chunk(yhat_unlabeled, args.heads, dim=-1)
        preds = torch.stack(yhat_unlabeled_chunked).softmax(-1)
        if args.reg_mode == "ratio":
            ratio_losses = (preds[:, :, 0].mean(-1) - unlabeled_zero_freq).abs()
            reg_loss = ratio_losses.mean()
        elif args.reg_mode == "entropy":
            avg_preds = preds.mean(1)
            entropies = -Categorical(probs=avg_preds).entropy()
            reg_loss = entropies.mean()
        elif "kl" in args.reg_mode:
            if "ratio" in args.reg_mode:
                avg_preds_source = torch.tensor(
                    [unlabeled_zero_freq, 1 - unlabeled_zero_freq]
                ).to(preds.device)
            else:
                avg_preds_source = (
                    torch.stack(yhat_chunked).softmax(-1).mean([0, 1]).detach()
                )
            avg_preds_target = preds.mean(1)
            dist_source = Categorical(probs=avg_preds_source)
            dist_target = Categorical(probs=avg_preds_target)
            if args.reg_mode in ["kl_forward", "kl_ratio_f"]:
                kl = torch.distributions.kl.kl_divergence(dist_source, dist_target)
            elif args.reg_mode in ["kl_backward", "kl_ratio_b"]:
                kl = torch.distributions.kl.kl_divergence(dist_target, dist_source)
            reg_loss = kl.mean()
        else:
            raise ValueError(f"{args.reg_mode=} not implemented!")
        loss_main += reg_loss * args.reg_weight

        if "bert" in args.model:
            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        else:
            optimizer.zero_grad()
            loss_main.backward()
            optimizer.step()

        del loss_main, repulsion_loss, reg_loss, all_loss_mains
        del yhat_chunked, yhat_unlabeled_chunked, preds

        if (batch_idx + 1) % log_every == 0:
            all_stats = {}
            for computer_idx, loss_computer in loss_computers.items():
                stats = loss_computer.get_stats()
                all_stats.update({f"{computer_idx}_{k}": v for k, v in stats.items()})
                loss_computer.reset_stats()
            csv_logger.log(epoch, batch_idx, all_stats)
            csv_logger.flush()

            worst_keys = [
                k for k in all_stats.keys() if "worst" in k and "alt" not in k
            ]
            worst_val = max([all_stats[k] for k in worst_keys])
            avg_keys = [
                k for k in all_stats.keys() if "group_avg_acc" in k and "alt" not in k
            ]
            avg_val = max([all_stats[k] for k in avg_keys])

            delta = (datetime.now() - args.start_time).total_seconds()
            N = len(loader) * args.n_epochs
            n = epoch * len(loader) + batch_idx
            delta_est = delta * N / n
            logger.write(
                f"Elapsed: {sec_to_str(delta)}/{sec_to_str(delta_est)} Epoch {epoch} batch {batch_idx+1}/{len(loader)}\t"
            )
            logger.write(f"Avg acc {avg_val*100:.1f}, Worst acc {worst_val*100:.1f}\n")
            logger.flush()


def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    log_every=50,
    scheduler=None,
):

    if is_training:
        model.train()
        if "bert" in args.model:
            model.zero_grad()
    else:
        model.eval()

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(loader):
            if args.debug and batch_idx > 3:
                break
            x, y, g, _ = batch
            y_cp, g_cp = copy.deepcopy(y), copy.deepcopy(g)
            del y, g
            y, g = y_cp, g_cp
            x, y, g = x.cuda(), y.cuda(), g.cuda()
            y_onehot = None

            outputs = model(x)

            loss_main = loss_computer.loss(
                outputs, y, g, is_training, y_onehot=y_onehot
            )

            if is_training:
                if "bert" in args.model:
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(
    model,
    criterion,
    dataset,
    logger,
    csv_loggers,
    args,
    epoch_offset,
):
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, dataset["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)

    # BERT uses its own scheduler and optimizer
    if "bert" in args.model:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon
        )

        length = len(dataset["train_loader"])
        t_total = length * args.n_epochs

        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=t_total
        )

    else:
        length = len(dataset["train_loader"])

        t_total = length * args.n_epochs

        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"{args.optimizer} not recognized")

        if args.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            )
        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_training_steps=t_total,
                num_warmup_steps=args.num_warmup_steps,
            )

            step_every_batch = True
            use_metric = False
        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=t_total,
                num_warmup_steps=args.num_warmup_steps,
            )
        elif args.scheduler == "StepLR":
            scheduler = StepLR(optimizer, t_total, gamma=args.step_gamma)
        else:
            scheduler = None

    if args.diversify:
        loss_computer_keys = [f"h{i}" for i in range(args.heads)] + [
            f"h{i}_alt" for i in range(args.heads)
        ]
        train_loss_computers = {
            k: LossComputer(
                args,
                criterion,
                is_robust=args.robust,
                dataset=dataset["train_data"],
                alpha=args.alpha,
                gamma=args.gamma,
                adj=adjustments,
                step_size=args.robust_step_size,
                normalize_loss=args.use_normalized_loss,
                btl=args.btl,
                min_var_weight=args.minimum_variational_weight,
            )
            for k in loss_computer_keys
        }
    else:
        train_loss_computer = LossComputer(
            args,
            criterion,
            is_robust=args.robust,
            dataset=dataset["train_data"],
            alpha=args.alpha,
            gamma=args.gamma,
            adj=adjustments,
            step_size=args.robust_step_size,
            normalize_loss=args.use_normalized_loss,
            btl=args.btl,
            min_var_weight=args.minimum_variational_weight,
        )

    best_val_acc = 0
    count = 1
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        train_kwargs = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "loader": dataset["train_loader"],
            "logger": logger,
            "csv_logger": csv_loggers["train"],
            "args": args,
            "is_training": True,
            "log_every": args.log_every,
            "scheduler": scheduler,
        }
        if args.diversify:
            run_epoch_divdis_train(
                loader_unlabeled=dataset["val_loader"],
                loss_computers=train_loss_computers,
                **train_kwargs,
            )
        else:
            run_epoch(loss_computer=train_loss_computer, **train_kwargs)

        logger.write(f"\nEpoch {epoch}, Validation:\n")
        with torch.no_grad():
            val_lc_kwargs = {
                "args": args,
                "criterion": criterion,
                "is_robust": args.robust,
                "step_size": args.robust_step_size,
                "alpha": args.alpha,
                "is_val": True,
            }
            val_run_kwargs = {
                "epoch": epoch,
                "model": model,
                "logger": logger,
                "args": args,
            }
            if args.diversify:
                val_loss_computers = {
                    k: LossComputer(**val_lc_kwargs, dataset=dataset["val_data"])
                    for k in loss_computer_keys
                }
                run_epoch_divdis_eval(
                    loss_computers=val_loss_computers,
                    loader=dataset["val_loader"],
                    csv_logger=csv_loggers["val"],
                    **val_run_kwargs,
                )
            else:
                val_loss_computer = LossComputer(
                    **val_lc_kwargs, dataset=dataset["val_data"]
                )
                run_epoch(
                    optimizer=optimizer,
                    loss_computer=val_loss_computer,
                    is_training=False,
                    loader=dataset["val_loader"],
                    csv_logger=csv_loggers["val"],
                    **val_run_kwargs,
                )

            if args.in_dist_testing:
                if args.diversify:
                    val_loss_computers = {
                        k: LossComputer(**val_lc_kwargs, dataset=dataset["val_id_data"])
                        for k in loss_computer_keys
                    }
                    run_epoch_divdis_eval(
                        loss_computers=val_loss_computers,
                        loader=dataset["val_id_loader"],
                        csv_logger=csv_loggers["val_id"],
                        **val_run_kwargs,
                    )
                else:
                    val_loss_computer = LossComputer(
                        **val_lc_kwargs, dataset=dataset["val_id_data"]
                    )
                    run_epoch(
                        optimizer=optimizer,
                        loss_computer=val_loss_computer,
                        is_training=False,
                        loader=dataset["val_id_loader"],
                        csv_logger=csv_loggers["val_id"],
                        **val_run_kwargs,
                    )

            if dataset["test_data"] is not None:
                logger.write(f"\nEpoch {epoch}, Testing:\n")
                test_lc_kwargs = {
                    "args": args,
                    "criterion": criterion,
                    "is_robust": args.robust,
                    "step_size": args.robust_step_size,
                    "alpha": args.alpha,
                }
                test_run_kwargs = {
                    "epoch": epoch,
                    "model": model,
                    "logger": logger,
                    "args": args,
                }
                if args.diversify:
                    test_loss_computers = {
                        k: LossComputer(**test_lc_kwargs, dataset=dataset["test_data"])
                        for k in loss_computer_keys
                    }
                    run_epoch_divdis_eval(
                        loss_computers=test_loss_computers,
                        loader=dataset["test_loader"],
                        csv_logger=csv_loggers["test"],
                        **test_run_kwargs,
                    )
                else:
                    test_loss_computer = LossComputer(
                        **test_lc_kwargs, dataset=dataset["test_data"]
                    )
                    run_epoch(
                        optimizer=optimizer,
                        loss_computer=test_loss_computer,
                        is_training=False,
                        loader=dataset["test_loader"],
                        csv_logger=csv_loggers["test"],
                        **test_run_kwargs,
                    )
                if args.in_dist_testing:
                    if args.diversify:
                        test_loss_computers = {
                            k: LossComputer(
                                **test_lc_kwargs, dataset=dataset["test_id_data"]
                            )
                            for k in loss_computer_keys
                        }
                        run_epoch_divdis_eval(
                            loss_computers=test_loss_computers,
                            loader=dataset["test_id_loader"],
                            csv_logger=csv_loggers["test_id"],
                            **test_run_kwargs,
                        )
                    else:
                        test_loss_computer = LossComputer(
                            **test_lc_kwargs, dataset=dataset["test_id_data"]
                        )
                        run_epoch(
                            optimizer=optimizer,
                            loss_computer=test_loss_computer,
                            is_training=False,
                            loader=dataset["test_id_loader"],
                            csv_logger=csv_loggers["test_id"],
                            **test_run_kwargs,
                        )

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss
                )
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss)  # scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, "%d_model.pth" % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, "last_model.pth"))

        if args.save_best:
            curr_val_acc = val_loss_computer.worst_group_acc

            logger.write(f"Current validation accuracy: {curr_val_acc}\n")
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, "best_model.pth"))
                logger.write(f"Best model saved at epoch {epoch}\n")

        if args.automatic_adjustment:
            gen_gap = (
                val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            )
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n"
                )
        logger.write("\n")
