import numpy as np
from detbackbone import Anchors
import tensorflow as tf
def calc_iou(a, b):
  # a(anchor) [boxes, (y1, x1, y2, x2)]
  # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]
  #print(b)
  area = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
  
  iw=np.minimum(a[:,np.newaxis,3],b[:,3])-np.maximum(a[:,np.newaxis,1],b[:,1])
  #print(iw)
  ih=np.minimum(a[:,np.newaxis,2],b[:,2])-np.maximum(a[:,np.newaxis,0],b[:,0])
  iw=np.clip(iw,0,None)
  ih=np.clip(ih,0,None)
  ua=(a[:,np.newaxis,2]-a[:,np.newaxis,0])*(a[:,np.newaxis,3]-a[:,np.newaxis,1])+area-iw*ih
  ua=np.clip(ua,1e-9,None)
  intersection = iw * ih
  #print(ua.shape,intersection.shape)
  IoU = intersection / ua
  #print(IoU,np.max(IoU,axis=1),np.argmax(IoU,axis=1))
  
  return IoU
def calc_y(anchors,annotations):
  batch_size=1
  classes=80
  anchor_widths = anchor[:, 3] - anchor[:, 1]
  anchor_heights = anchor[:, 2] - anchor[:, 0]
  anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
  anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights
  for j in range(batch_size):
    bbox_annotation = annotations[j]
    bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
    #print(bbox_annotation.shape)
    IoU = calc_iou(anchors[:, :], bbox_annotation[:, :4])
    Iou_max,Iou_argmax=np.max(IoU,axis=1),np.argmax(IoU,axis=1)
    positive_indices=Iou_max>=0.5
    num_positive_anchors=np.sum(positive_indices)
    assigned_annotations = bbox_annotation[Iou_argmax, :]
    targets=np.ones([anchors.shape[0],80])
    targets*=-1
    targets[Iou_max<0.4,:]=0
    targets[positive_indices, :] = 0
    targets[positive_indices, assigned_annotations[positive_indices, 4]] = 1 
    
    if positive_indices.sum() > 0:
      negative_indices=Iou_max<0.5
      assigned_annotations=assigned_annotations[:,:4]
      assigned_annotations[negative_indices,:]=0
      
      anchor_widths_pi = anchor_widths[positive_indices]
      anchor_heights_pi = anchor_heights[positive_indices]
      anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
      anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

      gt_widths = assigned_annotations[positive_indices, 2] - assigned_annotations[positive_indices, 0]
      gt_heights = assigned_annotations[positive_indices, 3] - assigned_annotations[positive_indices, 1]
      gt_ctr_x = assigned_annotations[positive_indices, 0] + 0.5 * gt_widths
      gt_ctr_y = assigned_annotations[positive_indices, 1] + 0.5 * gt_heights
      gt_widths = np.clip(gt_widths,1,None)
      gt_heights=np.clip(gt_heights,1,None)
      assigned_annotations[positive_indices,0]= (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
      assigned_annotations[positive_indices,1] = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
      assigned_annotations[positive_indices,2]= np.log(gt_widths / anchor_widths_pi)
      assigned_annotations[positive_indices,3] = np.log(gt_heights / anchor_heights_pi)
      print(targets.shape,assigned_annotations.shape)
    return targets,assigned_annotations
class FocalLoss():
    def __init__(self):
      self.anchors=Anchors(anchor_scale=4).get_boxes([512,512])
      self.alpha= 0.25
      self.gamma= 2
    def call(self,annotations,output):
      print(annotations,output)
      classifications=output[:,:,:80]
      regressions=output[:,:,80:]
     
      batch_size = output.shape[0]
      classification_losses = []
      regression_losses = []
      anchor_widths = self.anchors[:, 3] - self.anchors[:, 1]
      anchor_heights = self.anchors[:, 2] - self.anchors[:, 0]
      anchor_ctr_x = self.anchors[:, 1] + 0.5 * anchor_widths
      anchor_ctr_y = self.anchors[:, 0] + 0.5 * anchor_heights
      for j in range(batch_size):

        classification = classifications[j, :, :]
        regression = regressions[j, :, :]
        classification = tf.clip_by_value(classification,1e-4,1-1e-4)
        bbox_annotation = annotations[j]
        bbox_annotation = bbox_annotation[bbox_annotation[:, 4] !=-1]
        if bbox_annotation.shape[0] == 0:
          alpha_factor=tf.ones_like(classifications)*self.alpha
          alpha_factor = 1. - alpha_factor
          focal_weight = classification
          focal_weight = alpha_factor * tf.math.pow(focal_weight, self.gamma)
          bce = -(tf.math.log(1.0 - classification))
          cls_loss = focal_weight * bce
          regression_losses.append(tf.constant(0,dtype=self.anchors.dtype))
          classification_losses.append(np.sum(cls_loss))
          
          continue
        IoU = calc_iou(self.anchors[:, :], bbox_annotation[:, :4])
        Iou_max,Iou_argmax=np.max(IoU,axis=1),np.argmax(IoU,axis=1)
        targets = np.ones([self.anchors.shape[0],80],dtype=np.float32) * -1
        positive_indices=Iou_max>=0.5
        p_indices=np.where(Iou_max>=0.5)
        
        num_positive_anchors=np.sum(positive_indices)
        
        assigned_annotations = bbox_annotation[Iou_argmax, :]
        targets[Iou_max<0.4,:]=0
        targets[positive_indices, :] = 0
        targets[positive_indices, assigned_annotations[positive_indices, 4]] = 1 
        
        
        alpha_factor=tf.ones_like(classifications)*self.alpha
        alpha_factor = tf.where(targets==1, alpha_factor, 1. - alpha_factor)
        focal_weight = tf.where(targets==1, 1. - classification, classification)
        focal_weight = alpha_factor * tf.math.pow(focal_weight, self.gamma)
        #print(classification,targets)
        bce = -(tf.convert_to_tensor(targets) * tf.math.log(classification) + (1.0 - tf.convert_to_tensor(targets)) * tf.math.log(1.0 - classification))
    
        cls_loss = tf.convert_to_tensor(focal_weight) * bce
        zeros = tf.zeros_like(cls_loss)
        
        cls_loss = tf.where(tf.convert_to_tensor(targets!=-1), cls_loss, zeros)
        classification_losses.append(tf.reduce_sum(cls_loss) / np.clip(num_positive_anchors,1,None))
        if positive_indices.sum() > 0:
          assigned_annotations = assigned_annotations[positive_indices, :]

          anchor_widths_pi = anchor_widths[positive_indices]
          anchor_heights_pi = anchor_heights[positive_indices]
          anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
          anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

          gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
          gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
          gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
          gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

          # efficientdet style
          gt_widths = np.clip(gt_widths,1,None)
          gt_heights = np.clip(gt_heights, 1,None)
          
          targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
          targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
          targets_dw = np.log(gt_widths / anchor_widths_pi)
          targets_dh = np.log(gt_heights / anchor_heights_pi)
          targets_dx = targets_dx[:,np.newaxis]
          targets_dy = targets_dy[:,np.newaxis]
          targets_dw = targets_dw[:,np.newaxis]
          targets_dh = targets_dh[:,np.newaxis]
          targets = np.concatenate([targets_dx,targets_dy,targets_dw,targets_dh],axis=1)
           
          targets = targets.astype(np.float32)
          r=tf.concat([regression[tf.newaxis,i,:] for i in p_indices[-1]],axis=0)
          
          regression_diff =tf.math.abs(tf.convert_to_tensor(targets)-r)
          regression_loss=tf.where(regression_diff<=1./9.0,0.5 * 9.0 * tf.math.pow(regression_diff, 2),regression_diff - 0.5 / 9.0)
          regression_losses.append(tf.math.reduce_mean(regression_loss))
          #print(targets)
        
      return tf.math.reduce_mean(tf.stack(classification_losses),keepdims=True),tf.math.reduce_mean(tf.stack(regression_losses),keepdims=True)*50
if __name__=="__main__":
 reg=np.random.random([2,49104,84])
 reg=reg.astype(np.float32)
 classi=np.ones([2,110484,80])
 classi=classi.astype(np.float32)
 reg=tf.convert_to_tensor(reg)

 b=np.array([[[1,2,5,5,3],[5,5,8,8,4],[50,50,300,300,5]],[[0,0,90,90,44]]])
 
 #anchors=Anchors(anchor_scale=4)
 #anchor=anchors.get_boxes([768,768])
 
 #b*=200
 out=(classi,reg)
 loss=FocalLoss()
 l=loss.call(reg,b)
 
 #calc_y(anchor,b)
