from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm
import time
import torchattacks
from attack import PGD
from advertorch.attacks import PGDAttack
from torch import nn
from advertorch.context import ctx_noparamgrad_and_eval

from utils.config import opt
from data.voc_dataset import VOC_BBOX_LABEL_NAMES
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data.util import read_image
import copy

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000, flagadvtrain=False, adversary=None):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        # if flagadvtrain:
        #     scale = imgs[0].shape[2] / sizes[1]
        #     imgs, gt_bboxes_, gt_labels_ = imgs.cuda().float(), gt_bboxes_.cuda(), gt_labels_.cuda()
        #     imgs = adversary(imgs, gt_bboxes_, gt_labels_, scale)
        #     gt_bboxes_, gt_labels_ = gt_bboxes_.cpu(), gt_labels_.cpu()

        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def img2jpg(img, jpg_dir, img_suffix):
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype('uint8'))
    print(img)
    if not os.path.exists(jpg_dir):
        os.makedirs(jpg_dir)
    jpg_path = jpg_dir + img_suffix
    img.save(jpg_path, format='JPEG')
    jpg_img = read_image(jpg_path)
    return jpg_img


def add_bbox(ax, bbox, label, score):

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from {}'.format(opt.load_path))

    # trainer.vis.text(dataset.db.label_names, win='labels')
    adversary = None
    if opt.flagadvtrain:
        print("flagadvtrain turned: Adversarial training!")
        atk = PGD.PGD(trainer, eps=16/255, alpha=3/255, steps=4)
        # atk = torchattacks.PGD(trainer.faster_rcnn, eps=16, alpha=3, steps=4)
        # adversary = PGDAttack(trainer.faster_rcnn, loss_fn=nn.CrossEntropyLoss(), eps=16, nb_iter=4, eps_iter=3,
        #                       rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    best_map = 0
    lr_ = opt.lr
    normal_total_loss = []
    adv_total_loss = []
    total_time = 0.0
    total_imgs = 0
    true_imgs = 0
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        once = True
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            temp_img = copy.deepcopy(img).cuda()
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()

            if opt.flagadvtrain:
                before_time = time.time()
                img = atk(img, bbox, label, scale)
                after_time = time.time()
                # with ctx_noparamgrad_and_eval(trainer.faster_rcnn):
                #     img = adversary.perturb(img, label)
                # print("Adversarial training done!")

            total_time += after_time - before_time
            # print("Normal training starts\n")
            # trainer.train_step(img, bbox, label, scale)


            if (ii + 1) % opt.plot_every == 0:
                # adv_total_loss.append(trainer.get_meter_data()["total_loss"])
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                # trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                temp_ori_img_ = inverse_normalize(at.tonumpy(temp_img[0]))
                # img2jpg(temp_ori_img_, "imgs/orig_images/", "gt_img{}".format(ii))

                # temp_gt_img = visdom_bbox(temp_ori_img_,
                #                           at.tonumpy(bbox_[0]),
                #                           at.tonumpy(label_[0]))

                # plt.figure()
                # c, h, w = temp_gt_img.shape
                # plt.imshow(np.reshape(temp_gt_img, (h, w, c)))
                # plt.savefig("imgs/temp_orig_images/temp_gt_img{}".format(ii))
                # plt.close()

                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                # print("GT Label is {} and pred_label is {}".format(label_[0],))
                # img2jpg(ori_img_, "imgs/adv_images/", "adv_img{}".format(ii))

                # gt_img = visdom_bbox(ori_img_,
                #                      at.tonumpy(bbox_[0]),
                #                      at.tonumpy(label_[0]))

                # plt.figure()
                # c, h, w = gt_img.shape
                # plt.imshow(np.reshape(gt_img, (h, w, c)))
                # plt.savefig("imgs/orig_images/gt_img{}".format(ii))
                # plt.close()

                # trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)

                fig1 = plt.figure()
                ax1 = fig1.add_subplot(1,1,1)
                final1 = (at.tonumpy(img[0].cpu()).transpose(1,2,0).astype(int))
                ax1.imshow(final1)

                gt_img = visdom_bbox(ax1,at.tonumpy(_bboxes[0]),at.tonumpy(_labels[0]))
                fig1.savefig("imgs/adv_images/adv_img{}".format(ii))
                plt.close()

                # img2jpg(gt_img, "imgs/adv_images/", "adv_img{}".format(ii))

                _temp_bboxes, _temp_labels, _temp_scores = trainer.faster_rcnn.predict([temp_ori_img_], visualize=True)

                # temp_gt_img = visdom_bbox(at.tonumpy(temp_img[0].cpu()),
                #                           at.tonumpy(_temp_bboxes[0]),
                #                           at.tonumpy(_temp_labels[0]))

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(1, 1, 1)
                final2 = (at.tonumpy(temp_img[0].cpu()).transpose(1, 2, 0).astype(int))
                ax2.imshow(final2)

                gt_img = visdom_bbox(ax2, at.tonumpy(_bboxes[0]), at.tonumpy(_labels[0]))
                fig2.savefig("imgs/orig_images/gt_img{}".format(ii))
                plt.close()
                # img2jpg(temp_gt_img, "imgs/orig_images/", "gt_img{}".format(ii))

                # print("gt labels is {}, pred_orig_labels is {} and pred_adv_labels is {}".format(label_, _labels, _temp_labels))
                total_imgs += 1
                if len(_temp_labels) == 0:
                    continue
                if _labels[0].shape[0] == _temp_labels[0].shape[0] and (_labels[0] == _temp_labels[0]).all() is True:
                    true_imgs += 1
                # pred_img = visdom_bbox(ori_img_,
                #                        at.tonumpy(_bboxes[0]),
                #                        at.tonumpy(_labels[0]).reshape(-1),
                #                        at.tonumpy(_scores[0]))
                #

                # print("Shape of temp_orig_img_ is {}".format(temp_ori_img_.shape))
                # temp_pred_img = visdom_bbox(temp_ori_img_,
                #                             at.tonumpy(_temp_bboxes[0]),
                #                             at.tonumpy(_temp_labels[0]).reshape(-1),
                #                             at.tonumpy(_temp_scores[0]))
                #

                # trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                # trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        # fig = plt.figure()
        # ax1 = fig.add_subplot(2,1,1)
        # ax1.plot(normal_total_loss)
        # ax2 = fig.add_subplot(2,1,2)
        # ax2.plot(adv_total_loss)
        # fig.savefig("losses/both_loss{}".format(epoch))

        # eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num,
        #                    flagadvtrain=opt.flagadvtrain, adversary=atk)# adversary=adversary)

        # trainer.vis.plot('test_map', eval_result['map'])
        # lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        # log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
        #                                           str(eval_result['map']),
        #                                           str(trainer.get_meter_data()))
        # print(log_info)
        # # trainer.vis.log(log_info)
        #
        # if eval_result['map'] > best_map:
        #     best_map = eval_result['map']
        #     best_path = trainer.save(best_map=best_map)
        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
        #     lr_ = lr_ * opt.lr_decay

        if epoch == 0:
            break

        if epoch == 13:
            break

    print("Total time is {}".format(total_time))
    print("Avg time is {}".format(total_time/total_imgs))

    # plt.figure()
    # plt.plot(adv_total_loss)
    # plt.savefig("losses_adv_loss_final.jpg")
    # plt.close()


if __name__ == '__main__':
    import fire

    fire.Fire()
