import torch
import torch.nn as nn

import torch as t
from torchattacks import attack
from utils import array_tool as at
# from attack import attack

class PGD(attack.Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT: 0.3)
        alpha (float): step size. (DEFALUT: 2/255)
        steps (int): number of steps. (DEFALUT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        # >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=False)
        # >>> adv_images = attack(images, labels)

    """

    # def __init__(self, model, modeltrainer, eps=0.3, alpha=2 / 255, steps=40, random_start=False):
    #     super(PGD, self).__init__("PGD", model)
    #
    #     self.eps = eps
    #     self.alpha = alpha
    #     self.steps = steps
    #     self.random_start = random_start
    #     self.modeltrainer = modeltrainer

    # def forward(self, images, bboxes, labels, scale):
    #     r"""
    #     Overridden.
    #     """
    #     images = images.to(self.device)
    #     labels = labels.to(self.device)
    #     labels = self._transform_label(images, labels)
    #     loss = nn.CrossEntropyLoss()
    #
    #     adv_images = images.clone().detach()
    #
    #     if self.random_start:
    #         # Starting at a uniformly random point
    #         adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
    #         adv_images = torch.clamp(adv_images, min=0, max=1)
    #
    #
    #     for i in range(self.steps):
    #         adv_images.requires_grad = True
    #         _, _, H, W = adv_images.shape
    #         img_size = (H, W)
    #
    #         features = self.model.extractor(adv_images)
    #         # print(features.shape)
    #         # print(scale.shape)
    #
    #         rpn_locs, rpn_scores, rois, roi_indices, anchor = \
    #             self.model.rpn(features, img_size, scale)
    #
    #         # Since batch size is one, convert variables to singular form
    #         bbox = bboxes[0]
    #         label = labels[0]
    #         rpn_score = rpn_scores[0]
    #         rpn_loc = rpn_locs[0]
    #         roi = rois
    #
    #         sample_roi, gt_roi_loc, gt_roi_label = self.modeltrainer.proposal_target_creator(
    #             roi,
    #             at.tonumpy(bbox),
    #             at.tonumpy(label),
    #             self.modeltrainer.loc_normalize_mean,
    #             self.modeltrainer.loc_normalize_std)
    #
    #         gt_roi_label = at.totensor(gt_roi_label).long()
    #         sample_roi_index = t.zeros(len(sample_roi))
    #         _, roi_score = self.model.head(
    #             features,
    #             sample_roi,
    #             sample_roi_index)
    #         # outputs = self.model(adv_images)
    #
    #         cost = self._targeted*loss(roi_score, gt_roi_label.cuda()).to(self.device)
    #         # cost = self._targeted * loss(outputs[1], labels).to(self.device)
    #
    #         grad = torch.autograd.grad(cost, adv_images,
    #                                    retain_graph=True, create_graph=True,
    #                                    allow_unused=True)[0]
    #
    #         # print(grad)
    #         adv_images = adv_images.detach() + self.alpha * grad.sign()
    #         delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
    #         adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    #
    #     return adv_images

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)

        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, bboxes, labels, scale):
        r"""
        Overridden.
        """

        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=-1, max=1)


        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images, bboxes, labels, scale)
            (_, _, _, _, cost) = outputs
            # print("Shape of losses is {}".format(losses))

            # cost = self._targeted * loss(losses, labels).to(self.device)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=True, create_graph=True,
                                       allow_unused=True)[0]

            # print(grad)
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images