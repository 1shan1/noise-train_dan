#/usr/bin/python3
#-*- encoding=utf-8 -*-


import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from generator import NoisyImageGenerator, ValGenerator
from noise_model import get_noise_model


class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir_1", type=str, default='/data_2/noise2noise/train_dan/data/train_real.txt',
                        help="train image dir")
    parser.add_argument("--image_dir_2", type=str, default='/data_2/noise2noise/train_dan/data/train_fake.txt',
                        help="train image dir")
    parser.add_argument("--test_dir_1", type=str,default='/data_2/noise2noise/train_dan/data/val_real.txt',
                        help="test image dir")
    parser.add_argument("--test_dir_2", type=str,default='/data_2/noise2noise/train_dan/data/val_fake.txt',
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=24,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=600,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default='/data_2/noise2noise/weights.002-1315.793-17.37516.hdf5',
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="model",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for source images")
    parser.add_argument("--target_noise_model", type=str, default="gaussian,0,50",
                        help="noise model for target images")
    parser.add_argument("--val_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for validation source images")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    source_image_dir = args.image_dir_1
    target_image_dir = args.image_dir_2
    test_source_dir = args.test_dir_1
    test_target_dir = args.test_dir_2
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    model = get_model(args.model)

    if args.weight is not None:
        model.load_weights(args.weight)

    opt = Adam(lr=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    #source_noise_model = get_noise_model(args.source_noise_model)
    #target_noise_model = get_noise_model(args.target_noise_model)
    #val_noise_model = get_noise_model(args.val_noise_model)
    generator = NoisyImageGenerator(source_image_dir, target_image_dir, batch_size=batch_size,
                                    image_size=image_size)
    val_generator = ValGenerator(test_source_dir, test_target_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_loss",
                                     verbose=1,
                                     mode="min",
                                     save_best_only=False))

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
