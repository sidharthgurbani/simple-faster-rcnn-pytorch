from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

import torchattacks
from attack import PGD
from advertorch.attacks import PGDAttack
from torch import nn
from advertorch.context import ctx_noparamgrad_and_eval

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
import matplotlib.pyplot as plt
import numpy as np
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
    faster_rcnn_orig = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer_orig = FasterRCNNTrainer(faster_rcnn_orig).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        trainer_orig.load(opt.load_path)
        print('load pretrained model from {}'.format(opt.load_path))

    # trainer.vis.text(dataset.db.label_names, win='labels')
    adversary = None
    if opt.flagadvtrain:
        print("flagadvtrain turned: Adversarial training!")
        atk = PGD.PGD(trainer, eps=16, alpha=3, steps=4)
        # atk = torchattacks.PGD(trainer.faster_rcnn, eps=16, alpha=3, steps=4)
        # adversary = PGDAttack(trainer.faster_rcnn, loss_fn=nn.CrossEntropyLoss(), eps=16, nb_iter=4, eps_iter=3,
        #                       rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    best_map = 0
    lr_ = opt.lr
    normal_total_loss = []
    adv_total_loss = []
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        trainer_orig.reset_meters()
        once = True
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            temp_img = copy.deepcopy(img)

            trainer_orig.train_step(temp_img, bbox, label, scale)

            if opt.flagadvtrain:
                img = atk(img, bbox, label, scale)
                # with ctx_noparamgrad_and_eval(trainer.faster_rcnn):
                #     img = adversary.perturb(img, label)
                # print("Adversarial training done!")

            # print("Normal training starts\n")
            trainer.train_step(img, bbox, label, scale)

            normal_total_loss.append(trainer_orig.get_meter_data()["total_loss"])
            adv_total_loss.append(trainer.get_meter_data()["total_loss"])

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                # trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                temp_ori_img_ = inverse_normalize(at.tonumpy(temp_img[0]))
                temp_gt_img = visdom_bbox(temp_ori_img_,
                                          at.tonumpy(bbox_[0]),
                                          at.tonumpy(label_[0]))
                plt.figure()
                c, h, w = temp_gt_img.shape
                plt.imshow(np.reshape(temp_gt_img, (h, w, c)))
                plt.savefig("imgs/temp_orig_images/temp_gt_img{}".format(ii))
                plt.close()

                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                plt.figure()
                c, h, w = gt_img.shape
                plt.imshow(np.reshape(gt_img, (h, w, c)))
                plt.savefig("imgs/orig_images/gt_img{}".format(ii))
                plt.close()

                # trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes

                print("Shape of orig_img_ is {}".format(ori_img_.shape))
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))

                plt.figure()
                c, h, w = pred_img.shape
                plt.imshow(np.reshape(pred_img, (h, w, c)))
                plt.savefig("imgs/pred_images/pred_img{}".format(ii))
                plt.close()

                print("Shape of temp_orig_img_ is {}".format(temp_ori_img_.shape))
                _temp_bboxes, _temp_labels, _temp_scores = trainer_orig.faster_rcnn.predict([temp_ori_img_],
                                                                                            visualize=True)
                temp_pred_img = visdom_bbox(temp_ori_img_,
                                            at.tonumpy(_temp_bboxes[0]),
                                            at.tonumpy(_temp_labels[0]).reshape(-1),
                                            at.tonumpy(_temp_scores[0]))

                plt.figure()
                c, h, w = temp_pred_img.shape
                plt.imshow(np.reshape(temp_pred_img, (h, w, c)))
                plt.savefig("imgs/temp_pred_images/temp_pred_img{}".format(ii))
                plt.close()

                # trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                # trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                # trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(normal_total_loss)
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(adv_total_loss)
        fig.savefig("losses/both_loss{}".format(epoch))

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num,
                           flagadvtrain=opt.flagadvtrain, adversary=atk)# adversary=adversary)

        # trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        print(log_info)
        # trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break

        print("Best MAP is {}".format(best_map))


if __name__ == '__main__':
    import fire

    fire.Fire()
