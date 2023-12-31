"""
https://github.com/marrrcin/pytorch-resnet-mnist/blob/master/pytorch-resnet-mnist.ipynb
https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master/cifar10_models
"""
import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data import MNISTData
from module import MNISTModule


def main(args):

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(args.filepath, args.classifier, "{epoch:03d}"),
        monitor="acc/val",
        mode="max",
        # save_last=False,
        period=args.period,
        save_top_k=args.save_top_k,
        save_weights_only=True,
    )

    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        gpus=args.gpu_id,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint,
        precision=args.precision,

    )

    model = MNISTModule(args)
    data = MNISTData(args)
    trainloader = data.train_dataloader()
    data.save_train_data(trainloader, args.filepath)
    testloader = data.test_dataloader()
    data.save_test_data(testloader, args.filepath)

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, data)
        trainer.test()

if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="mlp")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--filepath", type=str, default="models")
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--save_top_k", type=int, default=-1)

    args = parser.parse_args()
    main(args)
