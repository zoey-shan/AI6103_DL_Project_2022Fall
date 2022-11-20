import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import argparse
from lightning_modules import UNetLightning, PhcDataModule, SkinCancerDataModule


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--channel', type=int, default=1, help='Number of channels')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--rfr', type=bool, default=False, help='Use Receptive Field Regularization')
    parser.add_argument('--rf_on_up', type=bool, default=False, help='Use Receptive Field Regularization on Upsampler')
    parser.add_argument('--rf_reg_weight', type=float, default=0.1, help='Receptive Field Regularization weight')
    parser.add_argument('--rf_reg_layers', type=int, default=4, help='Receptive Field Regularization layer numbers')
    parser.add_argument('--dataset', type=str, default='phc', help='Dataset to use')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.dataset=='skin_cancer':
        data_mod = SkinCancerDataModule(
            "/home/MSAI/yzhang253/datasets/skin lesion/trainx",
            "/home/MSAI/yzhang253/datasets/skin lesion/trainy",
            "/home/MSAI/yzhang253/datasets/skin lesion/validationx",
            "/home/MSAI/yzhang253/datasets/skin lesion/validationy",
            "/home/MSAI/yzhang253/datasets/skin lesion/testx",
            "/home/MSAI/yzhang253/datasets/skin lesion/testy",
            scale=0.5, mask_suffix='', 
            use_rf=args.rfr,
            batch_size=args.batch_size, num_workers=48
        )
    else:
        data_mod = PhcDataModule(
                images_dir='/home/MSAI/yzhang253/datasets/PhC_C2DH_U373/dev/img/', 
                masks_dir="/home/MSAI/yzhang253/datasets/PhC_C2DH_U373/dev/mask/",
                scale=0.5, mask_suffix='', 
                use_rf=args.rfr,
                batch_size=args.batch_size, num_workers=48
        )

    model = UNetLightning(
        use_rf=args.rfr,
        rf_on_up=args.rf_on_up,
        rf_reg_weight=args.rf_reg_weight,
        n_channels=args.channel,
        num_classes=args.classes,
        lr=args.lr,
        train_epochs=args.epochs,
        reg_layers=args.rf_reg_layers
    )
    trainer = Trainer(
        default_root_dir="lightning_logs/",
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=100,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    print("ARGS:", args)
    trainer.fit(model, data_mod)
    torch.save([model.tr_loss,model.va_loss,model.tr_acc,model.va_acc,model.tr_dice,model.va_dice,model.rf_reg_loss if args.rfr else []] ,
        'stats/stats_use_rf={}_on_up={}_rf_reg_weight={}_reg_layer={}_lr={}_epoch={}_dataset={}.pkl'.format(
            args.rfr, 
            args.rf_on_up,
            args.rf_reg_weight,
            args.rf_reg_layers,
            args.lr,
            args.epochs,
            args.dataset,
        ))