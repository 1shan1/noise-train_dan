import argparse
import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model
import os


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str,default='/data_2/noise2noise/train_dan/data/test/test_img',
                        help="test image dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, default='/data_2/noise2noise/train_dan/model/weights.077-5560.994-10.81238.hdf5',
                        help="trained weight file")
    parser.add_argument("--image_size", type=int, default=24,
                        help="training patch size")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default='/data_2/noise2noise/train_dan/data/test/out_test_img',
                        help="if set, save resulting images otherwise show result using imshow")
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    image_size = args.image_size
    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(Path(image_dir).glob("*.*"))

    for image_path in image_paths:
        img_name = os.path.basename(str(image_path))
        image = cv2.imread(str(image_path))
        #h, w, _ = image.shape
        #image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
        img = cv2.resize(image,(image_size,image_size))
        #img = image
        h, w, _ = img.shape

        out_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
        #noise_image = val_noise_model(image)
        pred = model.predict(np.expand_dims(img, 0))
        denoised_image = get_image(pred[0])
        out_image[:, :w] = img
        out_image[:, w:w * 2] = denoised_image
        #out_image[:, w * 2:] = denoised_image

        if args.output_dir:
            #cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
            cv2.imwrite(str(output_dir) +'/'+ img_name, denoised_image)
        else:
            cv2.imshow("result", out_image)
            key = cv2.waitKey(-1)
            # "q": quit
            if key == 113:
                return 0


if __name__ == '__main__':
    main()
