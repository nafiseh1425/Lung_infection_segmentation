#Overal infection segmentation
#Patient information error fixed
#Wrong size error fixed



from typing_extensions import Literal
import numpy as np
from numpy.core.fromnumeric import std
from numpy.lib.type_check import imag
from tensorflow import keras
from tensorflow.keras.models import load_model
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from PIL import ImageTk, Image  
from tkinter import filedialog 
from tkinter.ttk import * 
from pydicom import dcmread
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
import xlsxwriter
import pandas as pd
import glob
import segmentation_models as sm
from matplotlib import colors
from datetime import date, datetime
import segmentation_models as sm
from matplotlib import colors
from datetime import date
import torch
from torch import nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
import skimage.measure
from torch.utils.data import Dataset
import os
import sys
import SimpleITK as sitk
import pydicom as pyd
import logging
from tqdm import tqdm
import fill_voids
import skimage.morphology
import numpy as np
import os
from pydicom import dcmread




CMAP = colors.ListedColormap(['black','yellow', 'red'])
bounds=[0,0.5,1.5,2.5]
normm = colors.BoundaryNorm(bounds, CMAP.N)

import model as modellib
from config import Config

root = Tk() 
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT)
root.attributes('-fullscreen',FALSE)

L_lung = None 
L_lung_name = None
operator = ' '

def intro():
    global all_names, original_all_names, predictioncheck, directorycheck, Directory, Name, risk_scores, left_scores, right_scores, accuracy_scores, ABC_scores
    global intesnsities, pixel_counts, intesnsities_R, pixel_counts_R, intesnsities_L, pixel_counts_L, means, means_L, means_R, patient_ABC_score
    global load_directory,load_toggle,endcheck,predict_toggle,all_predict_toggle,std_T,std_L,std_R,patient_lung,patient_left_lung,patient_right_lung,patient_infection,patient_left_infection
    global patient_right_infection,patient_infection_conso,patient_left_infection_conso,patient_right_infection_conso
    global manual_whole,manual_left,manual_right,left_lung,right_lung,left_infection,right_infection,total_infection,total_lung,density,left_density,right_density
    global density_score,left_density_score,right_density_score,total_score,patient_left_score,patient_right_score
    global total_score_conso,patient_left_score_conso,patient_right_score_conso,total_score_ggo,patient_right_score_ggo,patient_left_score_ggo, prediction_counter

    Name = None
    Directory = "None"
    load_directory = "None"
    load_toggle = False
    endcheck = False
    predictioncheck = False
    directorycheck = False
    predict_toggle = False
    all_predict_toggle = False
    original_all_names=[]
    all_names=[]
    accuracy_scores = []
    ABC_scores = []
    patient_ABC_score = "No rating"
    risk_scores=[]
    left_scores=[]
    right_scores=[]

    intesnsities = [0]
    intesnsities_L = [0]
    intesnsities_R = [0]
    pixel_counts = [0]
    pixel_counts_L = [0]
    pixel_counts_R = [0]
    means = []
    means_L = []
    means_R = []
    std_T = 0
    std_L = 0
    std_R = 0

    patient_lung = float(0)
    patient_left_lung = float(0)
    patient_right_lung = float(0)
    patient_infection = float(0)
    patient_left_infection = float(0)
    patient_right_infection = float(0)
    patient_infection_conso = float(0)
    patient_left_infection_conso = float(0)
    patient_right_infection_conso = float(0)

    manual_whole = float(0)
    manual_left = float(0)
    manual_right = float(0)

    left_lung=np.zeros((512,512), dtype=bool)
    right_lung=np.zeros((512,512), dtype=bool)
    left_infection=np.zeros((512,512), dtype=bool)
    right_infection=np.zeros((512,512), dtype=bool)
    total_infection=np.zeros((512,512), dtype=bool)
    total_lung=np.zeros((512,512), dtype=bool)

    density=np.zeros((512,512,1), dtype=np.int16)
    left_density=np.zeros((512,512,1), dtype=np.int16)
    right_density=np.zeros((512,512,1), dtype=np.int16)

    density_score = 0
    left_density_score = 0
    right_density_score = 0

    total_score = float(0)
    patient_left_score = float(0)
    patient_right_score = float(0)

    total_score_conso = float(0)
    patient_left_score_conso = float(0)
    patient_right_score_conso = float(0)

    total_score_ggo = float(0)
    patient_right_score_ggo = float(0)
    patient_left_score_ggo = float(0)

    prediction_counter = 0

intro()

Model_2 = keras.models.load_model('Augmented_TverskyLoss.h5', compile=False)

Model_6 = keras.models.load_model('Augmented_BinaryCrossentropy_norm_every_slice_fixed.h5', compile=False)
Model_6.load_weights('Augmented_BinaryCrossentropy_norm_every_slice_fixed_best.hdf5')

Model_7 = keras.models.load_model('Augmented_binary_focal_loss_norm_every_slice_fixed_drop1.h5', compile=False)
Model_7.load_weights('Augmented_binary_focal_loss_norm_every_slice_fixed_drop1_best.hdf5')

Model_10 = keras.models.load_model('Augmented_10_times_TverskyLoss_norm_every_slice_fixed.h5', compile=False)
Model_10.load_weights('Augmented_10_times_TverskyLoss_norm_every_slice_fixed_best.hdf5')

Model_12 = keras.models.load_model('Augmented_10_times_BinaryCrossentropy_norm_every_slice_fixed_drop0.h5', compile=False)
Model_12.load_weights('Augmented_10_times_BinaryCrossentropy_norm_every_slice_fixed_drop0_best.hdf5')

Model_19 = keras.models.load_model('AttRes_2Data_TverskyLoss_Augmented_every_slice_fixed_D1.h5', compile=False)
Model_19.load_weights('AttRes_2Data_TverskyLoss_Augmented_every_slice_fixed_D1_best.hdf5')

# Model_T = sm.FPN("resnet34", encoder_weights=None, input_shape=(512, 512, 3), classes=3, activation='softmax')
# Model_T.load_weights('3_channel_best.hdf5')

# Model_p = sm.FPN("resnet34", encoder_weights=None, input_shape=(512, 512, 3), classes=3, activation='softmax')
# Model_p.load_weights('9k-Aug_150ep.hdf5')

#Model_p = keras.models.load_model('9k-Aug_150ep.hdf5', compile=False)

class LugConfig(Config):
    NAME = 'lug segmentation'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # background + 1 (lug)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 5
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    BACKBONE = 'resnet50'

    POST_NMS_ROIS_INFERENCE = 1000 
    POST_NMS_ROIS_TRAINING = 2000

    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3


config = LugConfig()
# config.display()

class InferenceConfig(LugConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #IMAGE_MIN_DIM = 512
    #IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.95
    
inference_config = InferenceConfig()

MODEL_DIR = "MRCNN/logs"
#model = modellib.MaskRCNN(mode="inference", config=inference_config,  model_dir=MODEL_DIR)
# model.load_weights(MODEL_DIR + r"\lug segmentation20211010T1809\mask_rcnn_lug segmentation_0018.h5", by_name=True)
#model.load_weights(MODEL_DIR + r"\lug segmentation20211010T1809\mask_rcnn_lug segmentation_0020.h5", by_name=True)
dicts = torch.load(MODEL_DIR +r'\lug segmentation20211010T1809\unet_r231covid-0de78a7e.pth', map_location=torch.device('cpu'))

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv', residual=False):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
            residual: if True, residual connections will be added
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            if i == 0 and residual:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual, first=True))
            else:
                self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
                                                    padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
                                            padding, batch_norm, residual))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        res = self.last(x)
        return self.softmax(res)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, residual=False, first=False):
        super(UNetConvBlock, self).__init__()
        self.residual = residual
        self.out_size = out_size
        self.in_size = in_size
        self.batch_norm = batch_norm
        self.first = first
        self.residual_input_conv = nn.Conv2d(self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

        if residual:
            padding = 1
        block = []

        if residual and not first:
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(in_size))

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))

        if not residual:
            block.append(nn.ReLU())
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            if self.in_size != self.out_size:
                x = self.residual_input_conv(x)
                x = self.residual_batchnorm(x)
            out = out + x

        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, residual=False):
        super(UNetUpBlock, self).__init__()
        self.residual = residual
        self.in_size = in_size
        self.out_size = out_size
        self.residual_input_conv = nn.Conv2d(self.in_size, self.out_size, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm2d(self.out_size)

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out_orig = torch.cat([up, crop1], 1)
        out = self.conv_block(out_orig)
        if self.residual:
            if self.in_size != self.out_size:
                out_orig = self.residual_input_conv(out_orig)
                out_orig = self.residual_batchnorm(out_orig)
            out = out + out_orig

        return out
def preprocess(img, label=None, resolution=[192, 192]):
    imgmtx = np.copy(img)
    lblsmtx = np.copy(label)

    imgmtx[imgmtx < -1024] = -1024
    imgmtx[imgmtx > 600] = 600
    cip_xnew = []
    cip_box = []
    cip_mask = []
    for i in range(imgmtx.shape[0]):
        if label is None:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], width=resolution[0], height=resolution[1])
        else:
            (im, m, box) = crop_and_resize(imgmtx[i, :, :], mask=lblsmtx[i, :, :], width=resolution[0],
                                           height=resolution[1])
            cip_mask.append(m)
        cip_xnew.append(im)
        cip_box.append(box)
    if label is None:
        return np.asarray(cip_xnew), cip_box
    else:
        return np.asarray(cip_xnew), cip_box, np.asarray(cip_mask)
def simple_bodymask(img):
    maskthreshold = -500
    oshape = img.shape
    img = ndimage.zoom(img, 128/np.asarray(img.shape), order=0)
    bodymask = img > maskthreshold
    bodymask = ndimage.binary_closing(bodymask)
    bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
    bodymask = ndimage.binary_erosion(bodymask, iterations=2)
    bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
    regions = skimage.measure.regionprops(bodymask.astype(int))
    if len(regions) > 0:
        max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
        bodymask = bodymask == max_region
        bodymask = ndimage.binary_dilation(bodymask, iterations=2)
    real_scaling = np.asarray(oshape)/128
    return ndimage.zoom(bodymask, real_scaling, order=0)
def crop_and_resize(img, mask=None, width=192, height=192):
    bmask = simple_bodymask(img)
    # img[bmask==0] = -1024 # this line removes background outside of the lung.
    # However, it has been shown problematic with narrow circular field of views that touch the lung.
    # Possibly doing more harm than help
    reg = skimage.measure.regionprops(skimage.measure.label(bmask))
    if len(reg) > 0:
        bbox = np.asarray(reg[0].bbox)
    else:
        bbox = (0, 0, bmask.shape[0], bmask.shape[1])
    img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
    if not mask is None:
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask = ndimage.zoom(mask, np.asarray([width, height]) / np.asarray(mask.shape), order=0)
        # mask = ndimage.binary_closing(mask,iterations=5)
    return img, mask, bbox
def reshape_mask(mask, tbox, origsize):
    res = np.ones(origsize) * 0
    resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
    imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
    res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
    return res
class LungLabelsDS_inf(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx, None, :, :].astype(np.float)
def read_dicoms(path, primary=True, original=True):
    allfnames = []
    for dir, _, fnames in os.walk(path):
        [allfnames.append(os.path.join(dir, fname)) for fname in fnames]

    dcm_header_info = []
    dcm_parameters = []
    unique_set = []  # need this because too often there are duplicates of dicom files with different names
    i = 0
    for fname in tqdm(allfnames):
        filename_ = os.path.splitext(os.path.split(fname)[1])
        i += 1
        if filename_[0] != 'DICOMDIR':
            try:
                dicom_header = pyd.dcmread(fname, defer_size=100, stop_before_pixels=True, force=True)
                if dicom_header is not None:
                    if 'ImageType' in dicom_header:
                        if primary:
                            is_primary = all([x in dicom_header.ImageType for x in ['PRIMARY']])
                        else:
                            is_primary = True

                        if original:
                            is_original = all([x in dicom_header.ImageType for x in ['ORIGINAL']])
                        else:
                            is_original = True

                        # if 'ConvolutionKernel' in dicom_header:
                        #     ck = dicom_header.ConvolutionKernel
                        # else:
                        #     ck = 'unknown'
                        if is_primary and is_original and 'LOCALIZER' not in dicom_header.ImageType:
                            h_info_wo_name = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID,
                                              dicom_header.ImagePositionPatient]
                            h_info = [dicom_header.StudyInstanceUID, dicom_header.SeriesInstanceUID, fname,
                                      dicom_header.ImagePositionPatient]
                            if h_info_wo_name not in unique_set:
                                unique_set.append(h_info_wo_name)
                                dcm_header_info.append(h_info)
                                # kvp = None
                                # if 'KVP' in dicom_header:
                                #     kvp = dicom_header.KVP
                                # dcm_parameters.append([ck, kvp,dicom_header.SliceThickness])
            except:
                logging.error("Unexpected error:", sys.exc_info()[0])
                logging.warning("Doesn't seem to be DICOM, will be skipped: ", fname)

    conc = [x[1] for x in dcm_header_info]
    sidx = np.argsort(conc)
    conc = np.asarray(conc)[sidx]
    dcm_header_info = np.asarray(dcm_header_info, dtype=object)[sidx]
    # dcm_parameters = np.asarray(dcm_parameters)[sidx]
    vol_unique = np.unique(conc, return_index=1, return_inverse=1)  # unique volumes
    n_vol = len(vol_unique[1])
    logging.info('There are ' + str(n_vol) + ' volumes in the study')

    relevant_series = []
    relevant_volumes = []

    for i in range(len(vol_unique[1])):
        curr_vol = i
        info_idxs = np.where(vol_unique[2] == curr_vol)[0]
        vol_files = dcm_header_info[info_idxs, 2]
        positions = np.asarray([np.asarray(x[2]) for x in dcm_header_info[info_idxs, 3]])
        slicesort_idx = np.argsort(positions)
        vol_files = vol_files[slicesort_idx]
        relevant_series.append(vol_files)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol_files)
        vol = reader.Execute()
        relevant_volumes.append(vol)

    return relevant_volumes
def get_input_image(path):
    if os.path.isfile(path):
        logging.info(f'Read input: {path}')
        input_image = sitk.ReadImage(path)
    else:
        logging.info(f'Looking for dicoms in {path}')
        dicom_vols = read_dicoms(path, original=False, primary=False)
        if len(dicom_vols) < 1:
            sys.exit('No dicoms found!')
        if len(dicom_vols) > 1:
            logging.warning("There are more than one volume in the path, will take the largest one")
        input_image = dicom_vols[np.argmax([np.prod(v.GetSize()) for v in dicom_vols], axis=0)]
    return input_image


def apply(image, model=None, force_cpu=True, batch_size=20, volume_postprocessing=True, noHU=False):
    if model is None:
        model = get_model('unet', 'R231')

    numpy_mode = isinstance(image, np.ndarray)
    if numpy_mode:
        inimg_raw = image.copy()
    else:
        inimg_raw = sitk.GetArrayFromImage(image)
        directions = np.asarray(image.GetDirection())
        if len(directions) == 9:
            inimg_raw = np.flip(inimg_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
    del image

    if force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.info("No GPU support available, will use CPU. Note, that this is significantly slower!")
            batch_size = 1
            device = torch.device('cpu')
    model.to(device)

    if not noHU:
        tvolslices, xnew_box = preprocess(inimg_raw, resolution=[256, 256])
        tvolslices[tvolslices > 600] = 600
        tvolslices = np.divide((tvolslices + 1024), 1624)
    else:
        # support for non HU images. This is just a hack. The models were not trained with this in mind
        tvolslices = skimage.color.rgb2gray(inimg_raw)
        tvolslices = skimage.transform.resize(tvolslices, [256, 256])
        tvolslices = np.asarray([tvolslices * x for x in np.linspace(0.3, 2, 20)])
        tvolslices[tvolslices > 1] = 1
        sanity = [(tvolslices[x] > 0.6).sum() > 25000 for x in range(len(tvolslices))]
        tvolslices = tvolslices[sanity]
    torch_ds_val = LungLabelsDS_inf(tvolslices)
    dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=batch_size, shuffle=False, pin_memory=False)

    timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

    with torch.no_grad():
        for X in tqdm(dataloader_val):
            X = X.float().to(device)
            prediction = model(X)
            pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
            timage_res = np.vstack((timage_res, pls))

    # postprocessing includes removal of small connected components, hole filling and mapping of small components to
    # neighbors
    if volume_postprocessing:
        outmask = postrocessing(timage_res)
    else:
        outmask = timage_res

    if noHU:
        outmask = skimage.transform.resize(outmask[np.argmax((outmask == 1).sum(axis=(1, 2)))], inimg_raw.shape[:2],
                                           order=0, anti_aliasing=False, preserve_range=True)[None, :, :]
    else:
        outmask = np.asarray(
            [reshape_mask(outmask[i], xnew_box[i], inimg_raw.shape[1:]) for i in range(outmask.shape[0])],
            dtype=np.uint8)

    if not numpy_mode:
        if len(directions) == 9:
            outmask = np.flip(outmask, np.where(directions[[0, 4, 8]][::-1] < 0)[0])

    return outmask.astype(np.uint8)

def postrocessing(label_image, spare=[]):
    '''some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. All connected components smaller than min_area will be removed
    '''

    # merge small components to neighbours
    regionmask = skimage.measure.label(label_image)
    origlabels = np.unique(label_image)
    origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        r_max_intensity = int(r.max_intensity)
        if r.area > origlabels_maxsub[r_max_intensity]:
            origlabels_maxsub[r_max_intensity] = r.area
            region_to_lobemap[r.label] = r_max_intensity

    for r in tqdm(regions):
        r_max_intensity = int(r.max_intensity)
        if (r.area < origlabels_maxsub[
            r_max_intensity] or r_max_intensity in spare) and r.area > 2:  # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n != spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto
            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                int(regions[regionlabels.index(mapto)].max_intensity)]:
                origlabels_maxsub[int(regions[regionlabels.index(mapto)].max_intensity)] += myarea
            regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[outmask_mapped == spare] = 0

    if outmask_mapped.shape[0] == 1:
        # holefiller = lambda x: ndimage.morphology.binary_fill_holes(x[0])[None, :, :] # This is bad for slices that show the liver
        holefiller = lambda x: skimage.morphology.area_closing(x[0].astype(int), area_threshold=64)[None, :, :] == 1
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask
def keep_largest_connected_component(mask):
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask



def bbox_3D(labelmap, margin=2):
    shape = labelmap.shape
    r = np.any(labelmap, axis=(1, 2))
    c = np.any(labelmap, axis=(0, 2))
    z = np.any(labelmap, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    rmin -= margin if rmin >= margin else rmin
    rmax += margin if rmax <= shape[0] - margin else rmax
    cmin, cmax = np.where(c)[0][[0, -1]]
    cmin -= margin if cmin >= margin else cmin
    cmax += margin if cmax <= shape[1] - margin else cmax
    zmin, zmax = np.where(z)[0][[0, -1]]
    zmin -= margin if zmin >= margin else zmin
    zmax += margin if zmax <= shape[2] - margin else zmax

    if rmax - rmin == 0:
        rmax = rmin + 1

    return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])


model = UNet(n_classes=3, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
model.load_state_dict(dicts)
model.eval()



def normalize_fixed(CT):
   Min=-1000
   Max=400
   CT=np.clip(CT, Min, Max)
   norm_data = (CT-Min)/(Max-Min)
   return norm_data   


def normalize(CT):
   Maxx, Minn = CT.max(), CT.min()
   norm_data = (CT-Minn)/(Maxx-Minn)
   return norm_data


def get_hu(Test):
   image = Test.pixel_array.astype(np.int16)

   intercept = Test.RescaleIntercept
   slope = Test.RescaleSlope

   if slope != 1:
      image = slope * image.astype(np.float64)
      image = image.astype(np.int16)

   image += np.int16(intercept)

   return np.array(image, dtype=np.int16)


def save_CT(name):
    global Directory
    slice = dcmread(Directory + '/' + name, force=True).pixel_array
    # print(slice.shape)
    if slice.shape[0] != 512 or slice.shape[1] != 512:
            slice = cv.resize(slice, (512,512), interpolation=cv.INTER_AREA)
    plt.imsave(Directory + f'/CT_images/{name}.png', slice, cmap='gray')


def get_files(directory):
    files = os.listdir(directory)
    files.sort(key= lambda x: (len (x), x))
    while files[0][0]=='.':
        del files[0]
    # print(files)    
    for file in files:
        # print(file)
        if file.startswith("Eval"):
            # print("removing: ", file)
            files.remove(file)
            # print(files)

    try:
        files.remove('CT_images')
    except:
        pass

    try:
        files.remove('corr_predicts')
    except:
        pass

    try:
        files.remove('lung')
    except:
        pass

    return files


def cleanfolder(directory):
   
   for file in glob.glob(directory + '/*'):
      os.remove(file)


def getdirectory():
    global all_names, original_all_names, predictioncheck, directorycheck, Directory, Name, risk_scores, left_scores, right_scores, accuracy_scores
    global intesnsities, pixel_counts, intesnsities_R, pixel_counts_R, intesnsities_L, pixel_counts_L, means, means_L, means_R, ABC_scores

    Directory = filedialog.askdirectory(title ='select folder')
    original_all_names = get_files(Directory)
    # all_names = get_files(Directory)
    risk_scores = [0] * len(original_all_names)
    left_scores = [0] * len(original_all_names)
    right_scores = [0] * len(original_all_names)
    accuracy_scores = [0] * len(original_all_names)
    ABC_scores = [0] * len(original_all_names)
    intesnsities = [0] * len(original_all_names)
    intesnsities_R = [0] * len(original_all_names)
    intesnsities_L = [0] * len(original_all_names)
    pixel_counts = [0] * len(original_all_names)
    pixel_counts_R = [0] * len(original_all_names)
    pixel_counts_L = [0] * len(original_all_names)
    means = [0] * len(original_all_names)
    means_R = [0] * len(original_all_names)
    means_L = [0] * len(original_all_names)
    Name = original_all_names[0]

    # print("final names are:" , original_all_names)



def center_view(xcord=730, ycord=70):
    global Name
    global L_center

    if load_toggle:
        img =Image.open( load_directory + "/CT_images/" + Name + ".png")
        img = img.resize((650,650), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img) 
        L_Center_name = Label(root, text= "Slice    "+ Name[1:] + '  ', font=("Calibri", 12))
        L_Center_name.place(x=xcord+300, y=ycord+10)

        global_center.config(image=img)
        global_center.image = img

        global_raw.config(image=img)
        global_raw.image = img

        patient_info()
        return

    if not os.path.exists(Directory + "/CT_images/" + Name + ".png"):
        save_CT(Name)
    # img = tk.PhotoImage(file = "CT_images/" + "I34" + ".png")
    img =Image.open(Directory + "/CT_images/" + Name + ".png")
    img = img.resize((650,650), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    # L_center = tk.Label(root, image = img)
    # L_center.image = img
    # L_center.place(x=xcord, y=ycord) 
    L_Center_name = Label(root, text= "Slice    "+ Name[1:] + '  ', font=("Calibri", 12))
    L_Center_name.place(x=xcord+300, y=ycord+10)

    global_center.config(image=img)
    global_center.image = img

    global_raw.config(image=img)
    global_raw.image = img

    patient_info()
    # patient_info_2()


def single_predict(xcord=450, ycord=100):
    global Name, Directory, left_lung, left_infection, right_lung, right_infection, total_infection, total_lung, predict_toggle, left_scores, right_scores, risk_scores, status_single
    global patient_lung, patient_infection, patient_left_lung, patient_right_lung, patient_left_infection, patient_right_infection, density_score, left_density_score, right_density_score
    global std_T, std_L, std_R, patient_infection_conso, patient_left_infection_conso, patient_right_infection_conso, prediction_counter

    left_lung=np.zeros((512,512), dtype=bool)
    right_lung=np.zeros((512,512), dtype=bool)
    # left_infection=np.zeros((512,512), dtype=bool)
    # right_infection=np.zeros((512,512), dtype=bool)
    # total_infection=np.zeros((512,512), dtype=bool)
    total_lung=np.zeros((512,512), dtype=bool)

    # Label(root, text= " "*40).place(x=1350, y=100)
    status_single.config(text="")

    try:
        if risk_scores[original_all_names.index(Name)] == "NI":
            status_single.config(text="No infection found.", foreground='red')

        if risk_scores[original_all_names.index(Name)] == "NL": 
            status_single.config(text="No lung found.", foreground='red')
    except:
        pass


    if not os.path.exists(Directory + "/corr_predicts/" + Name + ".png") and not os.path.exists(load_directory + "/corr_predicts/" + Name + ".png") and not load_toggle:

        if risk_scores[original_all_names.index(Name)] == "NI":
        #    Label(root, text= "No infection found.                  ", foreground='red').place(x=1350, y=100)
           status_single.config(text="No infection found.", foreground='red')
           center_view()
           root.update_idletasks()
           return
        if risk_scores[original_all_names.index(Name)] == "NL":
        #    Label(root, text= "No lung found.                  ", foreground='red').place(x=1350, y=100) 
           status_single.config(text="No lung found.", foreground='red')
           center_view()
           root.update_idletasks()
           return

        prediction_counter += 1

        # Label(root, text= "Working on it  ...                  ").place(x=1350, y=100)
        status_single.config(text="Working on it ...", foreground='black')
        root.update_idletasks()
        slice = dcmread(Directory + '/' + Name , force=True)
        raw_ct = get_hu(slice)
        if raw_ct.shape[0] != 512 or raw_ct.shape[1] != 512:
            raw_ct = cv.resize(raw_ct, (512,512), interpolation=cv.INTER_AREA)
        # print(raw_ct.min(), raw_ct.max())
        test_image=normalize(get_hu(slice))
        if test_image.shape[0] != 512 or test_image.shape[1] != 512:
            test_image = cv.resize(test_image, (512,512), interpolation=cv.INTER_AREA)
        # test_image_lung = normalize(slice.pixel_array)
        test_image_lung = cv.normalize(test_image , None , 0, 255, norm_type=cv.NORM_MINMAX)
        test_image_lung=np.expand_dims(test_image_lung, -1)
        test_image_lung = np.repeat(test_image_lung, 3, axis=-1)

        test_image_fixed=normalize_fixed(get_hu(slice))
        if test_image_fixed.shape[0] != 512 or test_image_fixed.shape[1] != 512:
            test_image_fixed = cv.resize(test_image_fixed, (512,512), interpolation=cv.INTER_AREA)
        # test_image=np.expand_dims(test_image, -1)
        test_image_fixed=np.expand_dims(test_image_fixed, -1)
        # test_image=np.expand_dims(test_image,0)
        test_image_fixed=np.expand_dims(test_image_fixed,0)
        # result_2=Model_2.predict(test_image)
        result_6=Model_6.predict(test_image_fixed)
        result_7=Model_7.predict(test_image_fixed)
        result_10=Model_10.predict(test_image_fixed)
        result_12=Model_12.predict(test_image_fixed)
        result_19=Model_19.predict(test_image_fixed)

        # test_image_T = np.repeat(test_image_fixed, 3, axis=-1)
        # result_T=Model_T.predict(test_image_T)
        # result_p=Model_p.predict(test_image_T)

        # result = np.squeeze(np.argmax(result_p, axis= -1))
        # result_p = np.squeeze(np.argmax(result_p, axis= -1))
        # print(np.unique(result_p))

        # plt.imshow(result_p, cmap=CMAP, norm=normm)
        # plt.show()

        # result = np.where(result_p==0, 0, 1)
        # plt.imshow(np.squeeze(result_T), cmap='gray')
        # plt.show()

        result = result_6 + result_7 + result_10 + result_12 + result_19
        result = np.where(np.squeeze(result)<0.5, 0, 1)

        # result_lung = model.detect([test_image_lung], verbose=1)
        result_lung = apply(get_input_image(Directory + '/' + Name), model, batch_size=1)
        result_lung = result_lung[0]

        left_lung = np.where(result_lung == 1, 0, result_lung)
        left_lung = np.where(left_lung == 2, 1, left_lung)

        right_lung = np.where(result_lung == 2, 0, result_lung)

        total_lung = left_lung + right_lung

        # plt.imshow(result_lung['masks'][...,1])
        # plt.show()

        # print(result_lung['masks'].shape)
        if len(np.unique(total_lung)) > 1:
        #if result_lung['masks'].shape[-1] > 1:
            # Label(root, text= " "*40).place(x=1350, y=100)
            #for i in range(result_lung['masks'].shape[-1]):
                #if result_lung['class_ids'][i] == 1:
                    #right_lung += result_lung['masks'][...,i]
                #else:
                    #left_lung += result_lung['masks'][...,i]

        
            total_lung = left_lung + right_lung
            left_infection = (left_lung * result).astype(bool)
            right_infection = (right_lung * result).astype(bool)
            total_infection = left_infection + right_infection

            # left_infection_p = (left_lung * result_p)
            # right_infection_p = (right_lung * result_p)
            # total_infection_p = (total_lung * result_p) 
            # print(np.unique(total_infection_p))

           
            # total_infection = np.where(raw_ct>-800, total_infection, 0)

            # print((total_infection*raw_ct).max())
            # print((total_infection*raw_ct).min())
            # print(np.unique(total_infection*raw_ct))

            # total_infection = result


            patient_lung += np.count_nonzero(total_lung)
            patient_left_lung += np.count_nonzero(left_lung)
            patient_right_lung += np.count_nonzero(right_lung)

            patient_infection += np.count_nonzero(total_infection)
            patient_left_infection += np.count_nonzero(left_infection)
            patient_right_infection += np.count_nonzero(right_infection)

            # patient_infection_conso += np.count_nonzero(np.where(total_infection_p==2,1,0))
            # patient_left_infection_conso += np.count_nonzero(np.where(left_infection_p==2,1,0))
            # patient_right_infection_conso += np.count_nonzero(np.where(right_infection_p==2,1,0))

            left_scores[original_all_names.index(Name)] = float(np.count_nonzero(left_infection)/(np.count_nonzero(left_lung)+1))*100
            right_scores[original_all_names.index(Name)] = float(np.count_nonzero(right_infection)/(np.count_nonzero(right_lung)+1))*100
            risk_scores[original_all_names.index(Name)] =  float(np.count_nonzero(total_infection)/(np.count_nonzero(total_lung)+1))*100

            pixel_counts[original_all_names.index(Name)] =  np.count_nonzero(total_infection)
            pixel_counts_L[original_all_names.index(Name)] =  np.count_nonzero(left_infection)
            pixel_counts_R[original_all_names.index(Name)] =  np.count_nonzero(right_infection)

            intesnsities[original_all_names.index(Name)] =  np.sum(raw_ct*total_infection)
            intesnsities_L[original_all_names.index(Name)] =  np.sum(raw_ct*left_infection)
            intesnsities_R[original_all_names.index(Name)] =  np.sum(raw_ct*right_infection)

            means[original_all_names.index(Name)] =  intesnsities[original_all_names.index(Name)]/(pixel_counts[original_all_names.index(Name)]+1)
            means_L[original_all_names.index(Name)] =  intesnsities_L[original_all_names.index(Name)]/(pixel_counts_L[original_all_names.index(Name)]+1)
            means_R[original_all_names.index(Name)] =  intesnsities_R[original_all_names.index(Name)]/(pixel_counts_R[original_all_names.index(Name)]+1)

            dummy_means = np.ma.masked_equal(means, 0)
            dummy_means_L = np.ma.masked_equal(means_L, 0)
            dummy_means_R = np.ma.masked_equal(means_R, 0)

            if np.unique(dummy_means).size > 1:
                std_T = np.std(dummy_means)
            
            if np.unique(dummy_means_L).size > 1:
                std_L = np.std(dummy_means_L)
      

            if np.unique(dummy_means_R).size > 1:
                std_R = np.std(dummy_means_R)

            # print(type(std_T))
            # print(std_L)
            # print(std_R)


            plt.imsave(Directory + '/lung/' + Name + '.png', total_lung)
            img1 = cv.imread(Directory + "/CT_images/" + Name + ".png")
            img3 = cv.imread(Directory + "/lung/" + Name + ".png")
            blended_2 = cv.addWeighted(img1, 0.7, img3, 0.3, 0)
            cv.imwrite(Directory + "/lung/" + Name + "_blended" + ".png", blended_2)

            if len(np.unique(total_infection)) > 1:
                # Label(root, text= " "*40).place(x=1350, y=100)
                # total_infection_p[0,0,...]=1
                # total_infection_p[0,-1,...]=2
                plt.imsave('predict.png', total_infection)
                img1 = cv.imread(Directory + "/CT_images/" + Name + ".png")
                img2 = cv.imread("predict.png")
                blended = cv.addWeighted(img1, 0.6, img2, 0.4, 0)
                cv.imwrite(Directory + '/corr_predicts/' + Name + '.png', blended)
                # Label(root, text= " "*40).place(x=1350, y=100)
                root.update_idletasks()

            else:
                # Label(root, text= "No infection found.                  ", foreground='red').place(x=1350, y=100)
                status_single.config(text="No infection found.", foreground='red')
                risk_scores[original_all_names.index(Name)] = "NI"
                predict_toggle = False
                return

        else:
            # Label(root, text= "No lung found.                  ", foreground='red').place(x=1350, y=100)
            status_single.config(text="No lung found.", foreground='red')
            risk_scores[original_all_names.index(Name)] = "NL"
            predict_toggle = False
            return

    try:
        if load_toggle:
            img =Image.open( load_directory + "/corr_predicts/" + Name + ".png")
        else:
            img =Image.open( Directory + "/corr_predicts/" + Name + ".png")
        img = img.resize((650,650), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        # L_center = tk.Label(root, image = img)
        # L_center.image = img
        # L_center.place(x=xcord, y=ycord) 
        # L_Center_name = Label(root, text= "Slice  "+ Name[1:], font=("Calibri", 12))
        # L_Center_name.place(x=xcord+320, y=ycord)

        center_view()

        global_center.config(image = img)
        global_center.image = img

    except:
        return

    if not load_toggle:
        try:
            density_score = np.sum(intesnsities)/(np.sum(pixel_counts)+1)
            left_density_score = np.sum(intesnsities_L)/(np.sum(pixel_counts_L)+1)
            right_density_score = np.sum(intesnsities_R)/(np.sum(pixel_counts_R)+1)
        except:
            pass

    status_single.config(text="")


def predict_all():
    global Name, Directory, left_lung, left_infection, right_lung, right_infection, total_infection, total_lung, predict_toggle, left_scores, right_scores, risk_scores
    global patient_lung, patient_infection, all_names, original_all_names, all_predict_toggle, patient_left_lung, patient_right_lung, patient_left_infection, patient_right_infection
    global density, left_density, right_density, density_score, left_density_score, right_density_score, std_L, std_R, std_T, means, means_L, means_R
    global patient_infection_conso,patient_left_infection_conso,patient_right_infection_conso
    global accuracy_scores,intesnsities,intesnsities_L,intesnsities_R,pixel_counts,pixel_counts_L,pixel_counts_R, ABC_scores
    
    patient_lung = float(0)
    patient_infection = float(0)
    patient_left_lung = float(0)
    patient_right_lung = float(0)
    patient_left_infection = float(0)
    patient_right_infection = float(0)
    patient_infection_conso = float(0)
    patient_left_infection_conso = float(0)
    patient_right_infection_conso = float(0)

    risk_scores = [0] * len(original_all_names)
    left_scores = [0] * len(original_all_names)
    right_scores = [0] * len(original_all_names)
    accuracy_scores = [0] * len(original_all_names)
    ABC_scores = [0] * len(original_all_names)
    intesnsities = [0] * len(original_all_names)
    intesnsities_R = [0] * len(original_all_names)
    intesnsities_L = [0] * len(original_all_names)
    pixel_counts = [0] * len(original_all_names)
    pixel_counts_R = [0] * len(original_all_names)
    pixel_counts_L = [0] * len(original_all_names)
    means = [0] * len(original_all_names)
    means_R = [0] * len(original_all_names)
    means_L = [0] * len(original_all_names)

    Label(root, text= "Working on it  ...                  ").place(x=650, y=5)
    root.update_idletasks()

    

    for N in original_all_names:
        
        left_lung=np.zeros((512,512), dtype=bool)
        right_lung=np.zeros((512,512), dtype=bool)
        left_infection=np.zeros((512,512), dtype=bool)
        right_infection=np.zeros((512,512), dtype=bool)
        total_infection=np.zeros((512,512), dtype=bool)
        total_lung=np.zeros((512,512), dtype=bool)

        slice = dcmread(Directory + '/' + N , force=True)
        raw_ct = get_hu(slice)
        result_lung = apply(get_input_image(Directory + '/' + N), model, batch_size=1)
        #save_CT(N)
        left_lung = np.squeeze(np.where(result_lung == 1, 0, result_lung))
        left_lung = np.squeeze(np.where(left_lung == 2, 1, left_lung))

        if left_lung.shape[0] != 512 or left_lung.shape[1] != 512:
            left_lung = cv.resize(left_lung, (512, 512), interpolation=cv.INTER_AREA)

        right_lung = np.squeeze(np.where(result_lung == 2, 0, result_lung))

        if right_lung.shape[0] != 512 or right_lung.shape[1] != 512:
            right_lung = cv.resize(right_lung, (512, 512), interpolation=cv.INTER_AREA)

        total_lung = left_lung + right_lung


        if raw_ct.shape[0] != 512 or raw_ct.shape[1] != 512:
            raw_ct = cv.resize(raw_ct, (512,512), interpolation=cv.INTER_AREA)
        test_image=normalize(get_hu(slice))

        if test_image.shape[0] != 512 or test_image.shape[1] != 512:
            test_image = cv.resize(test_image, (512,512), interpolation=cv.INTER_AREA)
        # test_image_lung = normalize(slice.pixel_array)
        test_image_lung = cv.normalize(test_image , None , 0, 255, norm_type=cv.NORM_MINMAX)
        test_image_lung=np.expand_dims(test_image_lung, -1)
        test_image_lung = np.repeat(test_image_lung, 3, axis=-1)

        test_image_fixed=normalize_fixed(get_hu(slice))
        if test_image_fixed.shape[0] != 512 or test_image_fixed.shape[1] != 512:
            test_image_fixed = cv.resize(test_image_fixed, (512,512), interpolation=cv.INTER_AREA)
        # test_image=np.expand_dims(test_image, -1)
        test_image_fixed=np.expand_dims(test_image_fixed, -1)
        # test_image=np.expand_dims(test_image,0)
        test_image_fixed=np.expand_dims(test_image_fixed,0)

        # test_image_T = np.repeat(test_image_fixed, 3, axis=-1)
        # result_p=Model_p.predict(test_image_T)

        # result_p = np.squeeze(np.argmax(result_p, axis= -1))
        # result = np.where(result_p==0, 0, 1)

        result_6=Model_6.predict(test_image_fixed)
        result_7=Model_7.predict(test_image_fixed)
        result_10=Model_10.predict(test_image_fixed)
        result_12=Model_12.predict(test_image_fixed)
        result_19=Model_19.predict(test_image_fixed)

        result = result_6 + result_7 + result_10 + result_12 + result_19
        result = np.where(np.squeeze(result)<0.5, 0, 1)

        #result_lung = model.detect([test_image_lung], verbose=1)
        #result_lung = result_lung[0]

        save_CT(N)
        if len(np.unique(total_lung))>1:

        #if result_lung['masks'].shape[-1] > 1:
            
            #for i in range(result_lung['masks'].shape[-1]):
                #if result_lung['class_ids'][i] == 1:
                    #right_lung += result_lung['masks'][...,i]
              
                #else:
                    #left_lung += result_lung['masks'][...,i]
                
        
            #total_lung = left_lung + right_lung
            left_infection = (left_lung * result).astype(bool)
            right_infection = (right_lung * result).astype(bool)
            total_infection = left_infection + right_infection

            # left_infection_p = (left_lung * result_p)
            # right_infection_p = (right_lung * result_p)
            # total_infection_p = (total_lung * result_p) 

            patient_lung += np.count_nonzero(total_lung)
            patient_left_lung += np.count_nonzero(left_lung)
            patient_right_lung += np.count_nonzero(right_lung)

            patient_infection += np.count_nonzero(total_infection)
            patient_left_infection += np.count_nonzero(left_infection)
            patient_right_infection += np.count_nonzero(right_infection)

            # patient_infection_conso += np.count_nonzero(np.where(total_infection_p==2,1,0))
            # patient_left_infection_conso += np.count_nonzero(np.where(left_infection_p==2,1,0))
            # patient_right_infection_conso += np.count_nonzero(np.where(right_infection_p==2,1,0))

            left_scores[original_all_names.index(N)] = float(np.count_nonzero(left_infection)/(np.count_nonzero(left_lung)+1))*100
            right_scores[original_all_names.index(N)] = float(np.count_nonzero(right_infection)/(np.count_nonzero(right_lung)+1))*100
            risk_scores[original_all_names.index(N)] =  float(np.count_nonzero(total_infection)/(np.count_nonzero(total_lung)+1))*100

            pixel_counts[original_all_names.index(N)] =  np.count_nonzero(total_infection)
            pixel_counts_L[original_all_names.index(N)] =  np.count_nonzero(left_infection)
            pixel_counts_R[original_all_names.index(N)] =  np.count_nonzero(right_infection)

            intesnsities[original_all_names.index(N)] =  np.sum(raw_ct*total_infection)
            intesnsities_L[original_all_names.index(N)] =  np.sum(raw_ct*left_infection)
            intesnsities_R[original_all_names.index(N)] =  np.sum(raw_ct*right_infection)

            means[original_all_names.index(N)] =  intesnsities[original_all_names.index(N)]/(pixel_counts[original_all_names.index(N)]+1)
            means_L[original_all_names.index(N)] =  intesnsities_L[original_all_names.index(N)]/(pixel_counts_L[original_all_names.index(N)]+1)
            means_R[original_all_names.index(N)] =  intesnsities_R[original_all_names.index(N)]/(pixel_counts_R[original_all_names.index(N)]+1)


            plt.imsave(Directory + '/lung/' + N + '.png', total_lung)
            img1 = cv.imread(Directory + "/CT_images/" + N + ".png")
            img3 = cv.imread(Directory + "/lung/" + N + ".png")
            blended_2 = cv.addWeighted(img1, 0.7, img3, 0.3, 0)
            cv.imwrite(Directory + "/lung/" + N + "_blended" + ".png", blended_2)

            if len(np.unique(total_infection)) > 1:
                
                # total_infection_p[0,0,...]=1
                # total_infection_p[0,-1,...]=2
                plt.imsave('predict.png', total_infection)
                img1 = cv.imread(Directory + "/CT_images/" + N + ".png")
                img2 = cv.imread("predict.png")
                blended = cv.addWeighted(img1, 0.6, img2, 0.4, 0)
                cv.imwrite(Directory + '/corr_predicts/' + N + '.png', blended)

            else:
                risk_scores[original_all_names.index(N)] = "NI"

        else:
            risk_scores[original_all_names.index(N)] = "NL"
    

    dummy_means = np.ma.masked_equal(means, 0)
    dummy_means_L = np.ma.masked_equal(means_L, 0)
    dummy_means_R = np.ma.masked_equal(means_R, 0)

    std_T = np.std(dummy_means)
    std_L = np.std(dummy_means_L)
    std_R = np.std(dummy_means_R)

    density_score = np.sum(intesnsities)/(np.sum(pixel_counts)+1)
    left_density_score = np.sum(intesnsities_L)/(np.sum(pixel_counts_L)+1)
    right_density_score = np.sum(intesnsities_R)/(np.sum(pixel_counts_R)+1)

    # print("Density Score: ", density_score)
    # print("Left Density Score: ", left_density_score)
    # print("Right Density Score: ", right_density_score)

    Label(root, text= "Done.                                  ").place(x=650, y=5)
    root.update_idletasks()
    all_predict_toggle = True
    predict_toggle = True
    single_predict()
    L_lung_view()
    risk_info()
    manual_assessment()
            
       
def L_lung_view():
    global Name, L_lung, L_lung_name, Directory
    if os.path.exists(Directory + "/lung/" + Name + "_blended" ".png") or os.path.exists(load_directory + "/lung/" + Name + "_blended" ".png"):
        if L_lung is not None:
            L_lung.destroy()

        if risk_scores[original_all_names.index(Name)] == "NL":
            return

        if load_toggle:
            img =Image.open( load_directory + "/lung/" + Name + "_blended" ".png")
        else:
            img =Image.open( Directory + "/lung/" + Name + "_blended" ".png" )

        img = img.resize((300,300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        L_lung = tk.Label(lung_frame, image = img)
        L_lung.image = img
        L_lung.pack()
        # L_lung_name = Label(root, text= "Lung detection")
        # L_lung_name.place(x=300, y=480)

    else:
        if L_lung is not None:
            L_lung.destroy()
            

def B_directory():
    global predict_toggle, all_predict_toggle, load_toggle, patient_left_infection, patient_right_infection, std_T, std_L, std_R,density_score,left_density_score,right_density_score
    global patient_lung, patient_infection, patient_infection_conso, patient_left_infection_conso, patient_right_infection_conso,patient_left_lung,patient_right_lung, status_single
    global Directory
    intro()
    getdirectory()
    status_single.config(text="")

    if not os.path.exists(Directory + "/CT_images"):
        os.makedirs(Directory + "/CT_images")
    if not os.path.exists(Directory + "/corr_predicts"):
        os.makedirs(Directory + "/corr_predicts")
    if not os.path.exists(Directory + "/lung"):
        os.makedirs(Directory + "/lung")

    # cleanfolder(Directory + "/CT_images")
    # cleanfolder(Directory + "/predictions")
    # cleanfolder(Directory + "/corr_predicts")
    # cleanfolder(Directory + "/lung")
    predict_toggle = False
    all_predict_toggle = False
    load_toggle = False
    center_view()
    scale = tk.Scale(root, orient= HORIZONTAL, length=650, command=scale_bar, to=len(original_all_names)-1)
    scale.place(x= 730, y= 750)
    Button(root, text = "Prediction  ON/OFF",command = B_predict).place(x=980,y=820)
    Label(root, text= " "*30).place(x=650, y=5)
    #tk.Label(manual_frame, text=" "*20, foreground='green').grid(row=6, column=1, columnspan=2) #Clear score submission alert
    tk.Label(risk_frame, text=" "*20, foreground='green').grid(row=23, column=1, columnspan=2) #Clear saving alert

    risk_info()
    L_lung_view()
    manual_assessment()

    if os.path.exists(Directory + "/corr_predicts/" + Name + ".png"):
        single_predict()


def B_predict():
    global predict_toggle

    # Label(root, text= " "*40).place(x=1350, y=100)
    status_single.config(text="")
    root.update_idletasks()

    if predict_toggle:
        center_view()
        predict_toggle = False
    else:
        predict_toggle = True
        single_predict()

    # print("Lung: ", patient_lung)
    L_lung_view()
    risk_info()
        

def scale_bar(number):
    # Label(root, text= " "*40).place(x=1350, y=100)
    status_single.config(text="")
    root.update_idletasks()
    global Name, predict_toggle
    predict_toggle = False
    Name = original_all_names[int(number)]
    
    if all_predict_toggle or load_toggle or os.path.exists(Directory + "/corr_predicts/" + Name + ".png"):
        center_view()
        single_predict()
        predict_toggle = True
    else:
        center_view()
    L_lung_view()
    risk_info()
    manual_assessment()


def patient_info():
    global Name
    global patient_name

    if not load_toggle:
        slice = dcmread(Directory + '/' + Name, force=True)
        patient_name = str(slice["PatientName"].value)

        try:
            Label(patien_info_frame, text="Patient Name :           ").grid(row=0, column=0, sticky=W, padx=10)
            Label(patien_info_frame, text=slice["PatientName"].value).grid(row=0, column=2, sticky=E, padx=10)
            Label(patien_info_frame, text="                           ").grid(row=0, column=1)
            Label(patien_info_frame, text="Patient Sex :             ").grid(row=1, column=0, sticky=W, padx=10)
            Label(patien_info_frame, text=slice["PatientSex"].value).grid(row=1, column=2, sticky=E, padx=10)
            Label(patien_info_frame, text="Patient Age :             ").grid(row=2, column=0, sticky=W, padx=10)
            Label(patien_info_frame, text=slice["PatientAge"].value).grid(row=2, column=2, sticky=E, padx=10)

            Label(patien_info_frame, text="").grid(row=3, column=0, sticky=W, padx=10)
            Label(patien_info_frame, text="").grid(row=4, column=0, sticky=W, padx=10)
        except:
            pass
    else:
        Label(patien_info_frame, text="Patient Name :           ").grid(row=0, column=0, sticky=W, padx=10)  
        Label(patien_info_frame, text=patient_name).grid(row=0, column=2, sticky=E, padx=10)

    Label(patien_info_frame, text="Total Slices :").grid(row=5, column=0, sticky=W, padx=10)  
    # Label(patien_info_frame, text="                           ").grid(row=4, column=1)
    Label(patien_info_frame, text=len(original_all_names)).grid(row=5, column=2)
    
    Label(patien_info_frame, text="Predicted Slices :").grid(row=6, column=0, sticky=W, padx=10) 

    if all_predict_toggle or load_toggle:
        Label(patien_info_frame, text=' ' + str(len(original_all_names)) + '  ').grid(row=6, column=2)
    else: 
        Label(patien_info_frame, text=' ' + str(prediction_counter) + '  ').grid(row=6, column=2)

# def patient_info_2():
#     Label(patien_info_frame, text="Total Slices :").grid(row=4, column=0, sticky=W, padx=10)  
#     # Label(patien_info_frame, text="                           ").grid(row=4, column=1)
#     Label(patien_info_frame, text=len(original_all_names)).grid(row=4, column=2, sticky=E)
    
#     Label(patien_info_frame, text="Predicted Slices :").grid(row=5, column=0, sticky=W, padx=10)  
#     Label(patien_info_frame, text=prediction_counter).grid(row=5, column=2, sticky=E)

def accuracy_func(event):
    global accuracy_scores, accuracy_options
    accuracy_scores[original_all_names.index(Name)] = accuracy_options["values"].index(accuracy_options.get())

def ABC_func(event):
    global ABC_scores, ABC_options
    ABC_scores[original_all_names.index(Name)] = ABC_options["values"].index(ABC_options.get())

def patient_ABC_func(event):
    global patient_ABC_score, patient_ABC_options
    patient_ABC_score = patient_ABC_options.get()
    

def risk_info():
    global Name, left_scores, right_scores, risk_scores, accuracy_options, accuracy_label, patient_left_score, patient_right_score, total_score, load_toggle
    global ratings, std_T, std_L, std_R, ABC_ratings
    global total_score_conso, patient_left_score_conso, patient_right_score_conso, total_score_ggo, patient_left_score_ggo, patient_right_score_ggo, ABC_label, ABC_options

    Label(risk_frame, text="Slice Left Infection").grid(row=0, column=0, padx=40)
    # # Label(risk_frame, text="").grid(row=0, column=1)  
    Label(risk_frame, text="Slice Right infection").grid(row=0, column=1, padx=40)
    Label(risk_frame, text=f"{left_scores[original_all_names.index(Name)]: .1f}"+ " % ", background='white').grid(row=1, column=0)
    Label(risk_frame, text=f"{right_scores[original_all_names.index(Name)]: .1f}"+ " % ", background='white').grid(row=1, column=1)
    ##Label(risk_frame, text="   ").grid(row=2, column=0, columnspan=2)
    Label(risk_frame, text="Total Slice Infection").grid(row=2, column=0, columnspan=2, pady=5)
    try:
        Label(risk_frame, text=f"{risk_scores[original_all_names.index(Name)]: .1f}" + " % ", background='white').grid(row=3, column=0, columnspan=2)
    except:
        Label(risk_frame, text="0.0 % ", background='white').grid(row=3, column=0, columnspan=2)

    Label(risk_frame, text="   ", font=('calibre',16, 'bold')).grid(row=5, column=0, columnspan=2)
    Label(risk_frame, text="              ", font=('calibre',16, 'bold')).grid(row=6, column=0)
    #Label(risk_frame, text="Prediction Accuracy").grid(row=6, column=1, columnspan=2, pady=5)

    #accuracy = tk.StringVar()
    #accuracy_options = ttk.Combobox(risk_frame, width = 27, textvariable = accuracy)
    #ratings = ["No rating", "1", "2", "3", "4", "5"]
    #accuracy_options["values"] = ("No rating", "1-Poor", "2", "3", "4", "5-Excelent")
    # accuracy_options.grid(row=7, column=0, columnspan=2, pady=5)
    #accuracy_options.current()
    #accuracy_options.bind('<<ComboboxSelected>>', accuracy_func)
    #accuracy_options.grid(row=7, column=1, columnspan=2, pady=5)

    #accuracy_label.grid(row=8, column=1, columnspan=2, pady=5)


    #Label(risk_frame, text="   ", font=('calibre',28, 'bold')).grid(row=9, column=1, columnspan=2)
    #Label(risk_frame, text="Predominant Lesion Type of this Slice").grid(row=10, column=1, columnspan=2, pady=5)
    #ABC = tk.StringVar()
    #ABC_options = ttk.Combobox(risk_frame, width = 10, textvariable = ABC)
    #ABC_ratings = ["No rating", "A", "B", "C"]
    #ABC_options["values"] = ABC_ratings
    #ABC_options.current()
    #ABC_options.bind('<<ComboboxSelected>>', ABC_func)
    #ABC_options.grid(row=11, column=1, columnspan=2, pady=5)
    #ABC_label.grid(row=12, column=1, columnspan=2, pady=5)

    #if not accuracy_scores[original_all_names.index(Name)]:
        #accuracy_label.config(text = "No rating yet.")
    #else:
       # accuracy_label.config(text = f"Your rating was: {ratings[accuracy_scores[original_all_names.index(Name)]]}")

    #if not ABC_scores[original_all_names.index(Name)]:
    #    ABC_label.config(text = "No rating yet.")
    #else:
     #   ABC_label.config(text = f"Your rating was: {ABC_ratings[ABC_scores[original_all_names.index(Name)]]}")

    # accuracy_options.bind('<<ComboboxSelected>>', accuracy_func)

    if not load_toggle:
        try:
            total_score = (patient_infection/(patient_lung+1))*100
        except:
            total_score = float(0)
        
        try:
            patient_left_score = (patient_left_infection/(patient_left_lung+1))*100
        except:
            patient_left_score = float(0)

        try:
            patient_right_score = (patient_right_infection/(patient_right_lung+1))*100
        except:
            patient_right_score = float(0)

    if not load_toggle:
        try:
            total_score_conso = (patient_infection_conso/(patient_lung+1))*100
        except:
            total_score_conso = float(0)

        try:
            patient_left_score_conso = (patient_left_infection_conso/(patient_left_lung+1))*100
        except:
            patient_left_score_conso = float(0)
        
        try:
            patient_right_score_conso = (patient_right_infection_conso/(patient_right_lung+1))*100
        except:
            patient_right_score_conso = float(0)
        
       
    if not load_toggle:
        total_score_ggo = total_score - total_score_conso
        patient_left_score_ggo = patient_left_score - patient_left_score_conso
        patient_right_score_ggo = patient_right_score - patient_right_score_conso
    
    
    Label(risk_frame, text="   ", font=('calibre',28, 'bold')).grid(row=9, column=0, columnspan=2)

    Label(risk_frame, text="Patient Total Infection", foreground='blue').grid(row=10, column=0, columnspan=2, pady=5)
    Label(risk_frame, text=f"    {total_score: .1f} %     Density(HU) : {int(density_score)}    " , background='white').grid(row=11, column=0, columnspan=2)
    #Label(risk_frame, text=f" GGO   {total_score_ggo: .1f} %  ", background='yellow').grid(row=12, column=0, columnspan=2)
    #Label(risk_frame, text=f" CONSO   {total_score_conso: .1f} %  ", background='orange').grid(row=13, column=0, columnspan=2)
    Label(risk_frame, text=f" Density STD:  {std_T: .1f}    " ).grid(row=14, column=0, columnspan=2, pady=5)

    Label(risk_frame, text="   ", font=('calibre',12, 'bold')).grid(row=15, column=0, columnspan=2)

    Label(risk_frame, text="Patient Left Infection", foreground='blue').grid(row=16, column=0, pady=5)
    Label(risk_frame, text="Patient Right Infection", foreground='blue').grid(row=16, column=1, pady=5)
    Label(risk_frame, text=f"    {patient_left_score: .1f} %     Density(HU) : {int(left_density_score)}    " , background='white').grid(row=17, column=0)
    Label(risk_frame, text=f"    {patient_right_score: .1f} %     Density(HU) : {int(right_density_score)}    ", background='white').grid(row=17, column=1)
    # Label(risk_frame, text=f" GGO   {patient_left_score_ggo: .1f} %  ", background='yellow', ).grid(row=18, column=0)
    # Label(risk_frame, text=f" GGO   {patient_right_score_ggo: .1f} %  ", background='yellow').grid(row=18, column=1)
    # Label(risk_frame, text=f" CONSO   {patient_left_score_conso: .1f} %  ", background='orange').grid(row=19, column=0)
    # Label(risk_frame, text=f" CONSO   {patient_right_score_conso: .1f} %  ", background='orange').grid(row=19, column=1)
    Label(risk_frame, text=f" Density STD:  {std_L: .1f}    ").grid(row=20, column=0, pady=5)
    Label(risk_frame, text=f" Density STD:  {std_R: .1f}    ").grid(row=20, column=1, pady=5)

    Label(risk_frame, text="   ", font=('calibre',20, 'bold')).grid(row=21, column=0, columnspan=2)

    Label(risk_frame, text="   ", font=('calibre',50, 'bold')).grid(row=21, column=1, columnspan=2)
    Button(risk_frame, text = "Save Results", command=write_excel).grid(row=22, column=0, columnspan=2)
    

def write_excel():
    global Name, patient_name, patient_left_score, patient_right_score, total_score, ABC_ratings, patient_ABC_score
    global std_T, std_L, std_R, density_score, left_density_score, right_density_score
    global patient_left_score_conso, patient_right_score_conso, total_score_conso
    global patient_left_score_ggo, patient_right_score_ggo, total_score_ggo, operator, ratings, Directory

    slice = dcmread(Directory + '/' + Name, force=True)
    patient_name = str(slice["PatientName"].value)

    # path = filedialog.askdirectory(title ='Save file path')
    if load_toggle:
        path = load_directory
        slice = dcmread(load_directory + '/' + Name, force=True)
        patient_name = str(slice["PatientName"].value)
    else:
        path = Directory
        slice = dcmread(Directory + '/' + Name, force=True)
        patient_name = str(slice["PatientName"].value)

    workbook = xlsxwriter.Workbook(f'{path}/Evaluationa_{patient_name}.xlsx', {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, patient_name)
    worksheet.write(1, 0, operator)
    worksheet.write(2, 0, "YYYY-MM-DD")
    worksheet.write(3, 0, str(date.today()))
    worksheet.write(4, 0, time.strftime("%H:%M:%S", time.localtime()))
    worksheet.write(0, 1, 'Left score')
    worksheet.write(0, 2, 'Right score')
    worksheet.write(0, 3, 'Total score')
    #worksheet.write(0, 4, 'Rating')
    worksheet.write(0, 4, 'Slice')
    #worksheet.write(0, 6, 'ABC')
    worksheet.write(0, 5, 'patient_left_score')
    worksheet.write(1, 5, patient_left_score)
    worksheet.write(0, 6, 'patient_right_score')
    worksheet.write(1, 6, patient_right_score)
    worksheet.write(0, 7, 'total_score')
    worksheet.write(1, 7, total_score)
    #worksheet.write(0, 10, 'assessment_left')
    #worksheet.write(1, 10, manual_left)
    #worksheet.write(0, 11, 'assessment_right')
    #worksheet.write(1, 11, manual_right)
    #worksheet.write(0, 12, 'assessment_whole')
    #worksheet.write(1, 12, manual_whole)
    worksheet.write(0, 8, 'Left_density')
    worksheet.write(1, 8, left_density_score)
    worksheet.write(0, 9, 'Right_density')
    worksheet.write(1, 9, right_density_score)
    worksheet.write(0, 10, 'Density')
    worksheet.write(1, 10, density_score)
    worksheet.write(0, 11, 'Left_std')
    worksheet.write(1, 11, std_L)
    worksheet.write(0, 12, 'Right_std')
    worksheet.write(1, 12, std_R)
    worksheet.write(0, 13, 'std')
    worksheet.write(1, 13, std_T)
    #worksheet.write(0, 19, 'Conso')
    #worksheet.write(1, 19, total_score_conso)
    #worksheet.write(0, 20, 'Left_conso')
    #worksheet.write(1, 20, patient_left_score_conso)
    #worksheet.write(0, 21, 'Right_conso')
    #worksheet.write(1, 21, patient_right_score_conso)
    #worksheet.write(0, 22, 'GGO')
    #worksheet.write(1, 22, total_score_ggo)
    #worksheet.write(0, 23, 'Left_GGO')
    #worksheet.write(1, 23, patient_left_score_ggo)
    #worksheet.write(0, 24, 'Right_GGO')
    #worksheet.write(1, 24, patient_right_score_ggo)
    #worksheet.write(0, 24, 'Patient_ABC')
    #worksheet.write(1, 24, patient_ABC_score)
    
    # print(original_all_names)
    for i in range(len(original_all_names)):
        worksheet.write(i+1, 1, left_scores[i])
        worksheet.write(i+1, 2, right_scores[i])
        worksheet.write(i+1, 3, risk_scores[i])
        #worksheet.write(i+1, 4, ratings[accuracy_scores[i]])
        # print(ratings[accuracy_scores[i]])
        worksheet.write(i+1, 5, original_all_names[i])
        #worksheet.write(i+1, 6, ABC_ratings[ABC_scores[i]])

    workbook.close()
    tk.Label(risk_frame, text="File saved!", foreground='green').grid(row=23, column=0, columnspan=2)
    

def submit_scores():
    global Name, left_entry, right_entry, whole_entry, manual_whole, manual_left, manual_right

    try:
        if float(whole_entry.get()) <= 100 and float(whole_entry.get()) >= 0:
            manual_whole = float(whole_entry.get())
    except:
        pass

    # try:
    #     if float(left_entry.get()) <= 100 and float(left_entry.get()) >= 0:
    #         manual_left = float(left_entry.get())
    # except:
    #     pass

    # try:
    #     if float(right_entry.get()) <= 100 and float(right_entry.get()) >= 0:
    #         manual_right = float(right_entry.get())
    # except:
    #     pass

    #tk.Label(manual_frame, text="Submitted!", foreground='green').grid(row=6, column=1, columnspan=2)


def manual_assessment():
    global left_entry, right_entry, whole_entry, patient_ABC_options
    
    #Label(manual_frame, text="Patient Total Infection in Percentage").grid(row=0, column=1, columnspan=2)
    # Label(manual_frame, text="Left Lung").grid(row=2, column=0, padx=65);  tk.Label(manual_frame, text="Right Lung").grid(row=2, column=1, padx=65)
    
    #whole_entry = tk.Entry(manual_frame)
    #whole_entry.grid(row=1, column=1, padx=10, columnspan=2, pady=15)
    # left_entry = tk.Entry(manual_frame)
    # left_entry.grid(row=3, column=0, padx=10)
    # right_entry = tk.Entry(manual_frame)
    # right_entry.grid(row=3, column=1, padx=10)

    #Label(manual_frame, text="Predominant Lesion Type").grid(row=2, column=1, columnspan=2)
    #patient_ABC = tk.StringVar()
    #patient_ABC_options = ttk.Combobox(manual_frame, width = 10, textvariable = patient_ABC)
    #ABC_ratings = ["No rating", "A", "B", "C"]
    #patient_ABC_options["values"] = ABC_ratings
    #patient_ABC_options.current()
    #patient_ABC_options.bind('<<ComboboxSelected>>', patient_ABC_func)
    #patient_ABC_options.grid(row=3, column=1, columnspan=2, pady=5)
    # patient_ABC_label.grid(row=12, column=1, columnspan=2, pady=5)

    #Label(manual_frame, text="                        ").grid(row=4, column=0)
    #Button(manual_frame, text = "Submit Evaluation", command=submit_scores).grid(row=7, column=1, columnspan=2)

def operator_name():
    global operator
    operator = name_entry.get()
    tk.Label(root, text="Submitted as:  " + operator + ' '*20, foreground='green').place(x=1510, y=1020)


def excel_reader(dir):
    global patient_name, left_scores, right_scores, risk_scores, original_all_names, patient_left_score, patient_right_score, total_score 
    global left_density_score, right_density_score, density_score, std_T, std_L, std_R, ABC_scores, accuracy_scores
    global patient_left_score_conso, patient_right_score_conso, total_score_conso
    global patient_left_score_ggo, patient_right_score_ggo, total_score_ggo, operator, ratings, Name

    df = pd.read_excel(glob.glob(dir + '/*.xlsx')[0], engine='openpyxl')
    # if patient_name == df.columns[0]:
    patient_name = df.columns[0]
    # operator = df.iloc[0, 0]
    left_scores = df.iloc[:,1].to_list()
    right_scores = df.iloc[:,2].to_list()
    risk_scores = df.iloc[:,3].to_list()
    #ratings = ["No rating", "1", "2", "3", "4" , "5"]
    #ABC_ratings = ["No rating", "A", "B", "C"]
    #accuracy_scores = [ratings.index(i) for i in df.iloc[:,4].to_list()]
    original_all_names = df.iloc[:,5].to_list()
    Name = original_all_names[0]
    #ABC_scores = [ABC_ratings.index(i) for i in df.iloc[:,6].to_list()]

    patient_left_score = df.iloc[0,7]
    patient_right_score = df.iloc[0,8]
    total_score = df.iloc[0,9]

    density_score = df.iloc[0,15]
    left_density_score = df.iloc[0,13]
    right_density_score = df.iloc[0,14]

    std_L = df.iloc[0,16]
    std_R = df.iloc[0,17]
    std_T = df.iloc[0,18]

    #total_score_conso = df.iloc[0,19]
    #patient_left_score_conso = df.iloc[0,20]
    #patient_right_score_conso = df.iloc[0,21]

    #total_score_ggo = df.iloc[0,22]
    #patient_left_score_ggo = df.iloc[0,23]
    #patient_right_score_ggo = df.iloc[0,24]


def load_data():
    global load_directory, load_toggle
    load_directory = filedialog.askdirectory(title ='select folder')    
    load_toggle = True
    excel_reader(load_directory)

    center_view()
    scale = tk.Scale(root, orient= HORIZONTAL, length=650, command=scale_bar, to=len(original_all_names)-1)
    scale.place(x= 730, y= 750)
    Button(root, text = "Prediction  ON/OFF",command = B_predict).place(x=980,y=820)
    Label(root, text= " "*30).place(x=500, y=5)
    #tk.Label(manual_frame, text=" "*20, foreground='green').grid(row=6, column=1, columnspan=2) #Clear score submission alert
    tk.Label(risk_frame, text=" "*20, foreground='green').grid(row=23, column=1, columnspan=2) #Clear saving alert

    manual_assessment()
    L_lung_view()
    risk_info()
    single_predict()


def multiple_folders():
    global all_names, original_all_names, predictioncheck, directorycheck, Directory, Name, risk_scores, left_scores, right_scores, accuracy_scores
    global intesnsities, pixel_counts, intesnsities_R, pixel_counts_R, intesnsities_L, pixel_counts_L, means, means_L, means_R, ABC_scores

    Directories = filedialog.askdirectory(title ='select a folder containing multiple patients')
    for folder_name in os.listdir(Directories):
        Directory = Directories + "/" + folder_name
        original_all_names = get_files(Directory)
        risk_scores = [0] * len(original_all_names)
        left_scores = [0] * len(original_all_names)
        right_scores = [0] * len(original_all_names)
        accuracy_scores = [0] * len(original_all_names)
        #ABC_scores = [0] * len(original_all_names)
        intesnsities = [0] * len(original_all_names)
        intesnsities_R = [0] * len(original_all_names)
        intesnsities_L = [0] * len(original_all_names)
        pixel_counts = [0] * len(original_all_names)
        pixel_counts_R = [0] * len(original_all_names)
        pixel_counts_L = [0] * len(original_all_names)
        means = [0] * len(original_all_names)
        means_R = [0] * len(original_all_names)
        means_L = [0] * len(original_all_names)
        Name = original_all_names[0]

        status_single.config(text="")

        if not os.path.exists(Directory + "/CT_images"):
            os.makedirs(Directory + "/CT_images")
        if not os.path.exists(Directory + "/corr_predicts"):
            os.makedirs(Directory + "/corr_predicts")
        if not os.path.exists(Directory + "/lung"):
            os.makedirs(Directory + "/lung")


        predict_all()
        scale = tk.Scale(root, orient= HORIZONTAL, length=650, command=scale_bar, to=len(original_all_names)-1)
        scale.place(x= 730, y= 750)
        Button(root, text = "Prediction  ON/OFF",command = B_predict).place(x=980,y=820)
        write_excel()
        root.update_idletasks()



Label(root, text= "Raw CT", foreground='green').place(x=350, y=50)
raw_frame = tk.Frame(root, height=650, width=650, relief=  "groove", borderwidth=5)
raw_frame.place(x= 50, y= 70)
raw_frame.grid_propagate (False)
raw_frame.pack_propagate (False)
global_raw = Label(raw_frame)
global_raw.pack()


Label(root, text= "Lung Segmentation", foreground='green').place(x=150, y=730)
lung_frame = tk.Frame(root, height=300, width=300, relief=  "groove", borderwidth=5)
lung_frame.place(x= 50, y= 750)
lung_frame.pack_propagate (False)

Label(root, text= "Patient Information", foreground='green').place(x=500, y=730)
patien_info_frame = tk.Frame(root, height=300, width=300, relief=  "groove", borderwidth=5)
patien_info_frame.place(x= 400, y= 750)
patien_info_frame.grid_propagate (False)

Label(root, text= "Infection Segmentation", foreground='green').place(x=1000, y=50)
CT_frame = tk.Frame(root, height=650, width=650, relief=  "groove", borderwidth=5)
CT_frame.place(x= 730, y= 70)
CT_frame.pack_propagate (False)
global_center = Label(CT_frame)
global_center.pack()


Label(root, text= "Slice-level Score", foreground='green').place(x=1550, y=50)
risk_frame = tk.Frame(root, height=650, width=380, relief=  "groove", borderwidth=5, pady=10)
risk_frame.place(x=1400,y=70)
risk_frame.grid_propagate (False)

#Label(root, text= "Operator's Evaluation", foreground='green').place(x=1520, y=730)
#manual_frame = tk.Frame(root, height=210, width=380, relief=  "groove", borderwidth=5, pady=10)
#manual_frame.place(x=1400,y=750)
#manual_frame.grid_propagate (False)

Button(root, text = "Directory",command = B_directory).place(x=100,y=0)
# Button(root, text = "Predict",command = single_predict).place(x=1300,y=150)

def predict_message():
    answer = tk.messagebox.askyesno(title='confirmation',
                    message='Are you sure you want to predict all slices?')
    if answer:
        predict_all()

Button(root, text = "Predict  All",command = predict_message).place(x=200,y=0)

Button(root, text = "Load  Predictions",command = load_data).place(x=300,y=0)

Button(root, text = "Predict  Multiple  Patients",command = multiple_folders).place(x=420,y=0)

Label(root, text= "Operator's Name :", foreground='black').place(x=1400, y=980)
name_entry = tk.Entry(root)
name_entry.place(x=1510,y=980)
Button(root, text = "Submit",command = operator_name).place(x=1700,y=980)

def exit():
    answer = tk.messagebox.askyesno(title='confirmation',
                    message='Please save evaluation before closing. Ready to close?')
    if answer:
        root.destroy()

Button(root, text = "close",command = exit).place(x=0,y=0)


#accuracy_label = Label(risk_frame, text= "", foreground='green')
#ABC_label = Label(risk_frame, text= "", foreground='green')
#patient_ABC_label = Label(risk_frame, text= "", foreground='green')
status_single = Label(root, text= " ")
status_single.place(x=1130, y=820)

root.mainloop() 