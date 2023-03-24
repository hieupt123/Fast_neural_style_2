import argparse
import os
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
import utils
# from transformer_net import TransformerNet
from transformer_net_2 import TransformerNet

from vgg import Vgg16
import sys
import random
import glob
from torchvision.utils import save_image

mean = [0.4763, 0.4507, 0.4094]
std = [0.2702, 0.2652, 0.2811]

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(model, optimizer, checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    current_epoch = checkpoint['current_epoch']
    start_batch_idx = checkpoint['start_batch_idx']
    return model, optimizer, current_epoch, start_batch_idx


def train(args):
    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load Image
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    # Create Transformer Network
    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    # load Style
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    name_style = args.style_image.split("/")[-1]
    name_style = name_style.split(".")[0]
    print(name_style)

    # Calculate features and gram matrix of style image
    features_style = vgg(style)
    gram_style = [utils.gram_matrix(y) for y in features_style]

    # Load check point
    if args.load_model:
        transformer, optimizer, current_epoch, start_batch_idx = load_checkpoint(transformer, optimizer,
                                                                                 torch.load(args.path_checkpoint_load))
    else:
        current_epoch = 0
        start_batch_idx = 0

    """ Get Sample """
    # image_samples = []
    # for path in glob.glob("/content/Fast_neural_style_2/images/content-images/*.jpg"):
    #     image_sample = utils.load_image(path, size=args.image_size)
    #     image_sample = style_transform(image_sample)
    #     image_samples += [image_sample]
    # image_samples = torch.stack(image_samples)
    for path in glob.glob("/content/Fast_neural_style_2/images/monalisa/*.jpg"):
        image_sample = utils.load_image(path)
        image_sample = style_transform(image_sample)
        image_sample = image_sample.unsqueeze(0).to(device)


    # def save_sample(epoch, batch_id):
    #     """ Evaluates the model and saves image samples """
    #     transformer.eval()
    #     with torch.no_grad():
    #         output = transformer(image_samples.to(device))
    #     image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
    #     path_save_image_test = os.path.join(args.checkpoint_model_dir, "test")
    #     if os.path.exists(path_save_image_test) == False:
    #         os.makedirs(path_save_image_test)
    #     save_image(image_grid, f"{path_save_image_test}/epoch_{epoch}_batch_id_{batch_id}.jpg", nrow=4)
    #     transformer.train()

    def save_sample(epoch, batch_id):
        """ Evaluates the model and saves image samples """
        transformer.eval()
        with torch.no_grad():
            output = transformer(image_sample.to(device))
        image = denormalize(output[0])
        path_save_image_test = os.path.join(args.checkpoint_model_dir, "test")
        if os.path.exists(path_save_image_test) == False:
            os.makedirs(path_save_image_test)
        save_image(image, f"{path_save_image_test}/epoch_{epoch}_batch_id_{batch_id}.jpg")
        transformer.train()

    ##################
    for epoch in range(current_epoch, args.epochs):
        transformer.train()
        count = 0
        for batch_idx, (x, _) in enumerate(train_loader):

            # set parameter of checkpoint
            if epoch == current_epoch and batch_idx < start_batch_idx:
                n_batch = len(x)
                count += n_batch
                continue

            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)
            y = y.to(device)

            # transformer.eval().cpu()
            # with torch.no_grad():
            #     for img in y:
            #         utils.save_image(f'transform/{i}.jpg', img.to('cpu'))
            #         i+=1
            # transformer.to(device).train()

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(train_dataset),
                                  content_loss.item(), style_loss.item(),
                                  total_loss.item()
                )
                save_sample(epoch,batch_idx)
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_idx + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = f"{name_style}_ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_idx + 1) + ".pth.tar"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                checkpoint = {'state_dict': transformer.state_dict(), 'optimizer': optimizer.state_dict(),
                              'current_epoch': epoch, 'start_batch_idx': batch_idx + 1,
                              'content_loss': content_loss.item(), 'style_loss': style_loss.item(),
                              'total_loss': total_loss.item()}
                print(f"Save chekpoint to: {ckpt_model_path}")
                save_checkpoint(checkpoint, ckpt_model_path)

                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = name_style + "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".pth"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if args.model.endswith(".onnx"):
        output = stylize_onnx(content_image, args)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(args.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            style_model.eval()
            if args.export_onnx:
                assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(
                    style_model, content_image, args.export_onnx, opset_version=11,
                ).cpu()
            else:
                output = style_model(content_image).cpu()
    utils.save_image(args.output_image, output[0])


def stylize_onnx(content_image, args):
    """
    Read ONNX model and run it using onnxruntime
    """

    assert not args.export_onnx

    import onnxruntime

    ort_session = onnxruntime.InferenceSession(args.model)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(content_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    return torch.from_numpy(img_out_y)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=8,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--load-model", type=int, required=True, default=0,
                                  help="load checkpoints de tiep tuc training hay khong")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--path-checkpoint-load", type=str, default=None,
                                  help="file checkpoint de load")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=100,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=100,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on cuda, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
