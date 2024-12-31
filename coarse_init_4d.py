import os
import shutil
import argparse
import numpy as np
import torch
import sys
import time
import json
# import pytorch3d.renderer as p3dr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from submodules.dust3r.dust3r.inference import inference
from submodules.dust3r.dust3r.model import AsymmetricCroCo3DStereo
from submodules.dust3r.dust3r.utils.device import to_numpy
from submodules.dust3r.dust3r.image_pairs import make_pairs
from submodules.dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images

def get_args_parser():
    parser = argparse.ArgumentParser(description="Coarse Initialization for 4D Reconstruction")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="Image size for input images")
    parser.add_argument("--model_path", type=str, default="submodules/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="Path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run the model on ('cuda' or 'cpu')")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--schedule", type=str, default='linear', help="Learning rate schedule")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--niter", type=int, default=300, help="Number of iterations for optimization")
    parser.add_argument("--focal_avg", action="store_true", help="Use average focal length")
    parser.add_argument("--llffhold", type=int, default=2, help="Hold out every Nth image")
    parser.add_argument("--n_views", type=int, default=75, help="Number of views to process")
    parser.add_argument("--img_base_path", type=str, default="/data/InstantSplat/collated_instantsplat_data/eval/dnerf/bouncingballs", help="Base path to the images")
    return parser

if __name__ == '__main__':
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Extract arguments
    model_path = args.model_path
    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    n_views = args.n_views
    img_base_path = args.img_base_path
    image_size = args.image_size
    
    # Prepare paths
    dustr_img_folder_path = os.path.join(img_base_path, f"dust3r_{n_views}_views_swin", "images")
    os.makedirs(dustr_img_folder_path, exist_ok=True)
    
    # Load the pre-trained model
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    
    ##### Load images ######
    # Load transformation data
    transform_json_path = os.path.join(img_base_path, "transforms_test.json")
    with open(transform_json_path, 'r') as f:
        transform_data = json.load(f)
    
    
    # Get list of image filenames
    train_img_list = sorted(os.listdir(os.path.join(img_base_path, "images")))
    print(f"Total images found: {len(train_img_list)}")
    # Apply llffhold to exclude every Nth image
    if args.llffhold > 0:
        train_img_list = [img_name for idx, img_name in enumerate(train_img_list) if (idx + 1) % args.llffhold != 0]
    print(f"Images after applying llffhold: {len(train_img_list)}")
    # Select the specified number of views
    indices = np.linspace(0, len(train_img_list) - 1, n_views, dtype=int)
    train_img_list = [train_img_list[i] for i in indices]
    print(f"Selected {len(train_img_list)} images for processing.")
    
    # Ensure the number of images matches the expected number of views
    assert len(train_img_list) == n_views, f"Number of images is not equal to {n_views}"
    
    # Copy selected images to the dust3r images folder if not already present
    existing_images = os.listdir(dustr_img_folder_path)
    if len(existing_images) != len(train_img_list):
        for img_name in train_img_list:
            shutil.copy(os.path.join(img_base_path, "images", img_name), os.path.join(dustr_img_folder_path, img_name))
    
    # Load images and resize them
    images, ori_size = load_images(dustr_img_folder_path, size=image_size)
    print(f"Original image size: {ori_size}")
    
    start_time = time.time()
    
    #### Make pairs & Inference ####
    pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=True)
    output = inference(pairs, model, args.device, batch_size=batch_size)
    output_colmap_path = dustr_img_folder_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)
    
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=niter, schedule=schedule, lr=lr, focal_avg=args.focal_avg)
    scene = scene.clean_pointcloud()
    
    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals() #shape: (10,1) so 10 views
    poses = to_numpy(scene.get_im_poses()) # shape: (10, 4, 4)
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())
    
    end_time = time.time()
    print(f"Time taken for {n_views} views: {end_time-start_time} seconds")
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)
    
    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    # print("pts_4_3dgs", pts_4_3dgs.shape) # -> selected points = (((512*512)-discarded points)*10, 3)
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    # print("color_4_3dgs", color_4_3dgs.shape) -> selected points = (((512*512)-discarded points)*10, 3)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    # print("pts_4_3dgs_all", pts_4_3dgs_all.shape) -> (512*512*10, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", focals.detach().cpu().numpy())