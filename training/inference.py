import torch
import warnings
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter("ignore")
import os
import argparse
import csv

from utils.config_utils import load_yaml
from vis_utils import ImgLoader

def build_model(pretrainewd_path: str,
                img_size: int, 
                fpn_size: int, 
                num_classes: int,
                num_selects: dict,
                use_fpn: bool = True, 
                use_selection: bool = True,
                use_combiner: bool = True, 
                comb_proj_size: int = None):
    from models.pim_module.pim_module_eval import PluginMoodel

    model = \
        PluginMoodel(img_size = img_size,
                     use_fpn = use_fpn,
                     fpn_size = fpn_size,
                     proj_type = "Linear",
                     upsample_type = "Conv",
                     use_selection = use_selection,
                     num_classes = num_classes,
                     num_selects = num_selects, 
                     use_combiner = use_combiner,
                     comb_proj_size = comb_proj_size)

    if pretrainewd_path != "":
        ckpt = torch.load(pretrainewd_path)
        model.load_state_dict(ckpt['model'])
    
    model.eval()

    return model

@torch.no_grad()
def sum_all_out(out, sum_type="softmax"):
    target_layer_names = \
    ['layer1', 'layer2', 'layer3', 'layer4',
    'FPN1_layer1', 'FPN1_layer2', 'FPN1_layer3', 'FPN1_layer4', 
    'comb_outs']

    sum_out = None
    for name in target_layer_names:
        if name != "comb_outs":
            tmp_out = out[name].mean(1)
        else:
            tmp_out = out[name]
        
        if sum_type == "softmax":
            tmp_out = torch.softmax(tmp_out, dim=-1)
        if sum_out is None:
            sum_out = tmp_out
        else:
            sum_out = sum_out + tmp_out # note that use '+=' would cause inplace error
    return sum_out


if __name__ == "__main__":
    # ===== 0. get setting =====
    parser = argparse.ArgumentParser("Visualize SwinT Large")
    parser.add_argument("-pr", "--pretrained_root", type=str, 
        help="contain {pretrained_root}/best.pt, {pretrained_root}/config.yaml")
    parser.add_argument("-ir", "--image_root", type=str)
    args = parser.parse_args()

    load_yaml(args, args.pretrained_root + "/config.yaml")

    # ===== 1. build model =====
    model = build_model(pretrainewd_path = args.pretrained_root + "/best.pt",
                        img_size = args.data_size, 
                        fpn_size = args.fpn_size, 
                        num_classes = args.num_classes,
                        num_selects = args.num_selects,
                        use_fpn = args.use_fpn, 
                        use_selection = args.use_selection,
                        use_combiner = args.use_combiner)
    model.cuda()

    # ===== 2. prediction =====
    output = [
        ["id", "label"],
    ]
    img_loader = ImgLoader(img_size = args.data_size)

    classes = os.listdir('./dataset/train')
    classes.sort()
    
    files = sorted(os.listdir(args.image_root))
    files = [f for f in files if f.lower() != 'desktop.ini']
    print(f"test: {len(files)} images")
    imgs = []
    img_names = []
    img_preds = []
    for fi, f in enumerate(files):
        img_path = args.image_root + "/" + f
        img, ori_img = img_loader.load(img_path)
        img = img.unsqueeze(0) # add batch size dimension
        imgs.append(img)
        img_names.append(f.split('.')[0])
        if (fi+1) % 32 == 0 or fi == len(files) - 1:    
            imgs = torch.cat(imgs, dim=0)
        else:
            continue
        with torch.no_grad():
            imgs = imgs.cuda()
            outs = model(imgs)
            sum_outs = sum_all_out(outs, sum_type="softmax") # softmax
            preds = torch.sort(sum_outs, dim=-1, descending=True)[1]
            for bi in range(preds.size(0)):
                img_preds.append(classes[int(preds[bi, 0])])
            
        for pair in zip(img_names, img_preds):
            print(list(pair))
            output.append(list(pair))
        imgs = []
        img_names = []
        img_preds = []

    with open("../output.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(output)

    print('done')