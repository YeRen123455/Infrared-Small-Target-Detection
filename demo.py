# Basic module

# Torch and visulization
from torchvision      import transforms

# Metric, loss .etc
from model.utils import *
from model.loss import *
from model.load_param_data import load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_ACM    import  ACM
from model.model_DNANet import  DNANet

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DNANet',
                        help='model name: DNANet,  ACM')

    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False (model==DNANet), False(model==ACM)')

    # parameter for ACM
    parser.add_argument('--blocks', type=int, default=3, help='multiple block')
    parser.add_argument('--fuse_mode', type=str, default='AsymBi', help='fusion mode')

    # data and pre-process
    parser.add_argument('--img_demo_dir', type=str, default='img_demo',
                        help='img_demo')
    parser.add_argument('--img_demo_index', type=str,default='target3',
                        help='target1, target2, target3')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    args = parser.parse_args()

    # the parser
    return args

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)
        img_dir   = args.img_demo_dir+'/'+args.img_demo_index+args.suffix

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        data            = DemoLoader (img_dir, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        img             = data.img_preprocess()

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'ACM':
            model       = ACM   (args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Load Checkpoint
        checkpoint      = torch.load('pretrain_DNANet_model.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        img = img.cuda()
        img = torch.unsqueeze(img,0)

        if args.deep_supervision == 'True':
            preds = self.model(img)
            pred  = preds[-1]
        else:
            pred  = self.model(img)
        save_Pred_GT_visulize(pred, args.img_demo_dir, args.img_demo_index, args.suffix)






def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





