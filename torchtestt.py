# python coarse_init_4d.py --img_base_path /home/InstantSplat/collated_instantsplat_data/eval/dnerf/bouncingballs --n_views 10 --focal_avg


# python train_joint_4d.py -s /home/InstantSplat/collated_instantsplat_data/eval/dnerf/bouncingballs/dust3r_10_views_swin -m ./output/eval/dnerf/bouncingballs/10_views_swin/ --n_views 10 --scene bouncingballs --iter 10000 --optim_pose


# python init_test_pose4d.py --img_base_path /home/InstantSplat/collated_instantsplat_data/eval/dnerf/bouncingballs --n_views 10 --focal_avg


# python render.py -s /home/InstantSplat/collated_instantsplat_data/eval/dnerf/bouncingballs/dust3r_10_views_swin -m ./output/eval/dnerf/bouncingballs/10_views_swin/ --n_views 10 --scene bouncingballs --optim_test_pose_iter 500 --iter 10000 --eval

# python metrics.py -m ./output/eval/dnerf/bouncingballs/10_views_swin/ --gt_pose_path /home/InstantSplat/collated_instantsplat_data/eval/dnerf/bouncingballs --iter 10000 --n_views 10
            
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_device()