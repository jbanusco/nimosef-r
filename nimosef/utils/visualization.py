import os
import nibabel as nib
import numpy as np


def label_to_color(segmentation, colormap=None):
    """
    Convert 2D segmentation map to RGB image.
    """
    if colormap is None:
        colormap = {
            0: [0, 0, 0],
            1: [255, 0, 0],
            2: [0, 255, 0],
            3: [0, 0, 255],
        }

    H, W = segmentation.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in colormap.items():
        mask = segmentation == label
        color_img[mask] = color
    return color_img


def compute_label_volumes(segmentation, labels, z_slice=None, to_ml=True, dvol=1):
    """
    Computes the volume for each label over time, with an option to restrict
    the analysis to a given slice (Z dimension).
    
    Parameters:
        segmentation (ndarray): 4D segmentation image of shape [X, Y, Z, T].
        labels (iterable): List of label values.
        z_slice (int, optional): If provided, restrict analysis to this slice index along Z.
    
    Returns:
        dict: keys are label values and values are lists containing the volume at each time point.
    """
    T = segmentation.shape[-1]
    volumes = {label: [] for label in labels}
    for t in range(T):
        if z_slice is not None:
            # Restrict analysis to the specified Z slice
            seg_t = segmentation[:, :, z_slice, t]
        else:
            # Use the entire 3D volume at time t
            seg_t = segmentation[..., t]
        for label in labels:
            vol = np.sum(seg_t == label)
            if to_ml:
                vol = (vol * dvol) / 1e3
            volumes[label].append(vol)
    return volumes


def get_imgs_experiment(img_folder, subject_id):
    # Get the path of the reconstructed segmentation
    im_path = os.path.join(img_folder, subject_id, f'{subject_id}_im_gt.nii.gz')
    seg_path = os.path.join(img_folder, subject_id, f'{subject_id}_seg_gt.nii.gz')

    # Predicted data
    pred_im_path = os.path.join(img_folder, subject_id, f'{subject_id}_rec.nii.gz')
    pred_seg_path = os.path.join(img_folder, subject_id, f'{subject_id}_seg.nii.gz')

    # Slice corrected predicted data
    pred_corr_im_path = os.path.join(img_folder, subject_id, f'{subject_id}_rec_corrected.nii.gz')
    pred_corr_seg_path = os.path.join(img_folder, subject_id, f'{subject_id}_seg_corrected.nii.gz')

    # Verify that all the files exist
    assert os.path.exists(im_path)
    assert os.path.exists(seg_path)
    assert os.path.exists(pred_im_path)
    assert os.path.exists(pred_seg_path)
    assert os.path.exists(pred_corr_im_path)
    assert os.path.exists(pred_corr_seg_path)

    # Load the images
    nii_im_gt = nib.load(im_path)
    nii_seg_gt = nib.load(seg_path)

    # Load the segmentation images
    nii_im_pred = nib.load(pred_im_path)
    nii_seg_pred = nib.load(pred_seg_path)
    nii_im_pred_corr = nib.load(pred_corr_im_path)
    nii_seg_pred_corr = nib.load(pred_corr_seg_path)

    # Compute the MSE between the images
    im_gt = nii_im_gt.get_fdata()
    im_pred = nii_im_pred.get_fdata()
    im_pred_corr = nii_im_pred_corr.get_fdata()

    # Get the segmentation
    seg_gt = nii_seg_gt.get_fdata()
    seg_pred = nii_seg_pred.get_fdata()
    seg_pred_corr = nii_seg_pred_corr.get_fdata()    

    # Dvol
    pix_resolution = np.linalg.norm(nii_im_gt.affine[:3, :3], axis=0)
    dvol = np.prod(pix_resolution)

    return im_gt, im_pred, im_pred_corr, seg_gt, seg_pred, seg_pred_corr, dvol


