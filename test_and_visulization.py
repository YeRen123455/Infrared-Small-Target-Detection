# Basic module
from tqdm             import tqdm
from model.parse_args_test import  parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# Model
from model.model_DNANet import  Res_CBAM_block
from model.model_ACM    import  ACM
from model.model_DNANet import  DNANet

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        # self.PD_FA = PD_FA(1,255)
        self.PD_FA = PD_FA(1,10)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DNANet':
            model       = DNANet(num_classes=1,input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'ACM':
            model       = ACM   (args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        checkpoint        = torch.load('D:\\Infrared-small-target\\code\\IR_detection_simple\\result\\' + args.model_dir)
        target_image_path = dataset_dir + '\\' +'visulization_result' + '\\' + args.st_model + '_visulization_result'
        target_dir        = dataset_dir + '\\' +'visulization_result' + '\\' + args.st_model + '_visulization_fuse'

        make_visulization_dir(target_image_path, target_dir)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                save_Pred_GT(pred, labels,target_image_path, val_img_ids, num, args.suffix)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)

                ture_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))
            test_loss = losses.avg
            scio.savemat(dataset_dir + '\\' +  'value_result'+ '\\' +args.st_model  + '_PD_FA_' + str(255),
                         {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            self.best_iou = mean_IOU
            save_result_for_test(dataset_dir, args.st_model,args.epochs, self.best_iou, recall, precision)


            source_image_path = dataset_dir + '\\images'
            if args.mode == 'TXT':
                txt_path = test_txt
                ids = []
                with open(txt_path, 'r') as f:
                    ids += [line.strip() for line in f.readlines()]

            for i in range(len(ids)):
                source_image = source_image_path + '\\' + ids[i] + args.suffix
                target_image = target_image_path + '\\' + ids[i] + args.suffix
                shutil.copy(source_image, target_image)
            for i in range(len(ids)):
                source_image = target_image_path + '\\' + ids[i] + args.suffix
                img = Image.open(source_image)
                img = img.resize((256, 256), Image.ANTIALIAS)
                img.save(source_image)
            # for m in range(len(ids)):
            #     plt.rcParams['font.sans-serif'] = ['STSong']  # 中文宋体
            #     plt.figure(figsize=(10, 6))
            #     plt.subplot(1, 3, 1)
            #     img = plt.imread(target_image_path +'\\'+ ids[m] +args.suffix)
            #     plt.imshow(img,cmap = 'gray')
            #     plt.xlabel("原始图像", size=11)
            #
            #     plt.subplot(1, 3, 2)
            #     img = plt.imread(target_image_path +'\\'+ ids[m] + '_GT'+args.suffix)
            #     plt.imshow(img,cmap = 'gray')
            #     plt.xlabel("真实结果", size=11)
            #
            #     plt.subplot(1, 3, 3)
            #     img = plt.imread(target_image_path +'\\'+ ids[m] + '_Pred'+args.suffix)
            #     plt.imshow(img,cmap = 'gray')
            #     plt.xlabel("实验结果", size=11)
            #
            #     plt.savefig(target_dir +'\\'+ ids[m].split('.')[0] + "_fuse"+args.suffix, facecolor='w', edgecolor='red')


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





