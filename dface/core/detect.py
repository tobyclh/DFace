import cv2
import time
import numpy as np
import torch
import torch.nn.functional as F
from dface.core.models import PNet,RNet,ONet
import dface.core.utils as utils
import dface.core.image_tools as image_tools
from torch.jit import script

def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):

    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        if(use_cuda):
            pnet.load_state_dict(torch.load(p_model_path))
            pnet.cuda()
        else:
            # forcing all GPU tensors to be in CPU while loading
            pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        if (use_cuda):
            rnet.load_state_dict(torch.load(r_model_path))
            rnet.cuda()
        else:
            rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        if (use_cuda):
            onet.load_state_dict(torch.load(o_model_path))
            onet.cuda()
        else:
            onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()
        
    return pnet,rnet,onet




class MtcnnDetector(object):
    """
        P,R,O net face detection and landmarks align
    """
    def __init__(self,
                 pnet = None,
                 rnet = None,
                 onet = None,
                 min_face_size=12,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709,
                 ):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.use_cuda = self.rnet_detector.use_cuda
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor
   
    # @script
    def generate_bounding_box(self, _map, reg, scale, threshold):
        """
            generate bbox from feature map
        Parameters:
        ----------
            map: torcn Tensor , 1 x n x m
                detect score for each position
            reg: torcn Tensor , 4 x n x m
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        cellsize = 12

        t_index = (_map > threshold).nonzero()
        t_index = t_index.t()
        # print(f't_index {t_index.shape}')
        if t_index.shape[0] > 0:
            dx1, dy1, dx2, dy2 = reg[0, 0, t_index[1], t_index[2]], reg[0, 1, t_index[1], t_index[2]], reg[0, 2, t_index[1], t_index[2]], reg[0, 3, t_index[1], t_index[2]]
            reg = torch.stack([dx1, dy1, dx2, dy2])

            score = _map[:, t_index[1], t_index[2]]
            t_index = t_index.float()
            boundingbox = torch.cat([((stride * t_index[2:2+1]) / scale),
                                    ((stride * t_index[1:1+1]) / scale),
                                    ((stride * t_index[2:2+1] + cellsize) / scale),
                                    ((stride * t_index[1:1+1] + cellsize) / scale),
                                    score,
                                    reg,
                                    # landmarks
                                    ])
            boundingbox = boundingbox.t()
        else: 
            # find nothing
            boundingbox = torch.Tensor([])
        return boundingbox



    
    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: torch Tensor
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """


        net_size = 12
        current_scale = float(net_size) / self.min_face_size    # find initial scale
        im_resized = resize_image(im, current_scale)
        _, _, current_height, current_width = im_resized.shape

        # fcn
        all_boxes = list()
        start_while_loop = time.time()

        while min(current_height, current_width) > net_size:
            feed_imgs = im_resized

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            cls_map, reg = self.pnet_detector(feed_imgs)
            cls_map, reg = cls_map.cpu(), reg.cpu()
            boxes = self.generate_bounding_box(cls_map[ 0, :, :], reg, current_scale, self.thresh[0])

            current_scale *= self.scale_factor
            
            im_resized = resize_image(im, current_scale)
            _, _, current_height, current_width = im_resized.shape

            if boxes.nelement() == 0:
                continue
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
        end_while_loop = time.time()
        print(f'end_while_loop : {end_while_loop - start_while_loop}')


        if len(all_boxes) == 0:
            return None, None
        all_boxes = torch.cat(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        # boxes = all_boxes[:, :5]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # landmark_keep = all_boxes[:, 9:].reshape((5,2))


        boxes = all_boxes[:,:5]

        # boxes = boxes.t()

        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = torch.stack([ align_topx,
                              align_topy,
                              align_bottomx,
                              align_bottomy,
                              all_boxes[:, 4],
                              ],dim=-1)
        # boxes_align = boxes_align.t()

        return boxes, boxes_align

    
    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: torch Tensor 1x3xHxW
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_align: numpy array
            boxes after calibration
        """
        _, _, h, w = im.shape

        if dets is None:
            return None,None

        dets = square_bbox(dets)
        dets[:, 0:4] = torch.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]

        '''
        # helper for setting RNet batch size
        batch_size = self.rnet_detector.batch_size
        ratio = float(num_boxes) / batch_size
        if ratio > 3 or ratio < 0.3:
            print "You may need to reset RNet batch size if this info appears frequently, \
        face candidates:%d, current batch_size:%d"%(num_boxes, batch_size)
        '''

        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = torch.zeros(1, 3, tmph[i], tmpw[i])
            tmp[..., dy[i]:edy[i]+1, dx[i]:edx[i]+1] = im[..., y[i]:ey[i]+1, x[i]:ex[i]+1]
            crop_im = F.interpolate(tmp, size=(24, 24))
            crop_im_tensor = crop_im
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = torch.cat(cropped_ims_tensors)

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.cpu()
        reg = reg.cpu()
        # landmark = landmark.cpu().data.numpy()


        keep_inds = (cls_map.squeeze() > self.thresh[1]).nonzero().squeeze()

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            _cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_cls = _cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        # keep_landmark = landmark[keep]


        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        boxes = torch.cat([ keep_boxes[:,0:4], keep_cls[:,0:1]], dim=-1)

        align_topx = keep_boxes[:,0] + keep_reg[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_reg[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_reg[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_reg[:,3] * bh

        boxes_align = torch.stack([align_topx,
                               align_topy,
                               align_bottomx,
                               align_bottomy,
                               keep_cls[:, 0],
                             ], dim=-1)

        return boxes, boxes_align

    
    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes_align: numpy array
            boxes after calibration
        landmarks_align: numpy array
            landmarks after calibration

        """
        _, _, h, w = im.shape

        if dets is None:
            return None, None

        dets = square_bbox(dets)
        dets[:, 0:4] = torch.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h, cuda=1 if self.use_cuda else 0)
        num_boxes = dets.shape[0]


        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = torch.FloatTensor(1, 3, tmph[i], tmpw[i]).fill_(0)
            tmp[..., dy[i]:edy[i]+1, dx[i]:edx[i]+1] = im[..., y[i]:ey[i]+1, x[i]:ex[i]+1]
            crop_im = F.interpolate(tmp, size=(48, 48))
            crop_im_tensor = crop_im
            # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = torch.cat(cropped_ims_tensors)

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg, landmark = self.onet_detector(feed_imgs)
        cls_map, reg, landmark = cls_map.cpu(), reg.cpu(), landmark.cpu()

        keep_inds = (cls_map.squeeze() > self.thresh[2]).nonzero().squeeze()

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            _cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = _cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]




        boxes_align = torch.stack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 ],dim=-1)

        landmark =  torch.stack([align_landmark_topx + keep_landmark[:, 0] * bw,
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 align_landmark_topx + keep_landmark[:, 8] * bw,
                                 align_landmark_topy + keep_landmark[:, 9] * bh,
                                 ], dim=-1)

        return boxes_align, landmark


    def detect_face(self,img):
        """Detect face over image"""
        boxes_align = torch.Tensor([])
        landmark_align = torch.Tensor([])

        img = image_tools.convert_image_to_tensor(img).unsqueeze(0)

        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return torch.Tensor([]), torch.Tensor([])
            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return torch.Tensor([]), torch.Tensor([])
            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return torch.Tensor([]), torch.Tensor([])
            t3 = time.time() - t
            t = time.time()
            print("time cost " + '{:.3f}'.format(t1+t2+t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2, t3))

        return boxes_align, landmark_align




@script
def resize_image(img, scale):
    # type: (Tensor, float) -> Tensor
    """
    resize image and transform dimention to [batchsize, channel, height, width]
    Parameters:
    ----------
        img: torch Tensor , BxCxHxW

        scale: float number
            scale factor of resize operation
    Returns:
    -------
        transformed image tensor , 1 x channel x height x width
    """
    _, _, height, width = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_height, new_width)
    img_resized = F.interpolate(img, size=new_dim, mode='bilinear', align_corners=True)
    return img_resized


@script
def square_bbox(bbox):
    """
        convert bbox to square
    Parameters:
    ----------
        bbox: torch Tensor , shape n x m
            input bbox
    Returns:
    -------
        square bbox
    """
    square_bbox = bbox

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    
    l = torch.max(h, w)

    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5

    square_bbox[:, 2] = square_bbox[:, 0] + l - 1
    square_bbox[:, 3] = square_bbox[:, 1] + l - 1
    return square_bbox



def pad(bboxes, w, h, cuda=0):
    """
        pad the the boxes
    Parameters:
    ----------
        bboxes: torch Tensor, N x 5
            input bboxes
        w: float number
            width of the input image
        h: float number
            height of the input image
    Returns :
    ------
        dy, dx : torch Tensor, n x 1
            start point of the bbox in target image
        edy, edx : torch Tensor, n x 1
            end point of the bbox in target image
        y, x : torch Tensor, n x 1
            start point of the bbox in original image
        ex, ex : torch Tensor, n x 1
            end point of the bbox in original image
        tmph, tmpw: torch Tensor, n x 1
            height and width of the bbox
    """

    tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).float()
    tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).float()
    numbox = bboxes.shape[0]

    dx = torch.zeros(numbox)
    dy = torch.zeros(numbox)

    edx, edy  = tmpw.clone()-1, tmph.clone()-1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = (ex > w-1).nonzero()
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = (ey > h-1).nonzero()
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = (x < 0).nonzero()
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = (y < 0).nonzero()
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy.int(), edy.int(), dx.int(), edx.int(), y.int(), ey.int(), x.int(), ex.int(), tmpw.int(), tmph.int()]
    # return_list = [item.int() for item in return_list]

    return return_list