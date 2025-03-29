import os
import os.path as osp
from pathlib import Path

# import torch
from loguru import logger
from pycocotools.coco import COCO
import numpy as np

from trackers.ocsort_tracker.ocsort import OCSort

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

class Predictor:
    def __init__(self, args):
        self.ann_coco = COCO(args.ann_path)
        self.pred_coco = self.ann_coco.loadRes(args.pred_path)
        self.FILENAME2IMGID = {}
        img_ids = self.ann_coco.getImgIds()
        for img_id in img_ids:
            img_info = self.ann_coco.loadImgs(img_id)[0]
            self.FILENAME2IMGID[img_info["file_name"]] = img_id

    def inference(self, video_image):
        img_id = self.FILENAME2IMGID[video_image]
        img_info = self.ann_coco.loadImgs(img_id)[0]
        anns = self.pred_coco.loadAnns(self.pred_coco.getAnnIds(imgIds=img_id))
        if len(anns) == 0:
            pred = [None]
            return pred, img_info
        
        pred = np.ndarray((len(anns), 5))
        for idx, ann in enumerate(anns):
            pred[idx, 0] = ann["bbox"][0]
            pred[idx, 1] = ann["bbox"][1]
            pred[idx, 2] = ann["bbox"][0] + ann["bbox"][2]
            pred[idx, 3] = ann["bbox"][1] + ann["bbox"][3]
            pred[idx, 4] = ann["conf" if "conf" in ann.keys() else "score"]
        return [pred], img_info

def get_video_image_dict(root_path):
    video_image_dict = {}
    
    for video_name in os.listdir(root_path):
        video_path = osp.join(root_path, video_name)
        
        if not osp.isdir(video_path):
            continue

        image_names = []
        for maindir, _, file_name_list in os.walk(video_path):
            for filename in file_name_list:
                ext = osp.splitext(filename)[1].lower()
                if ext in IMAGE_EXT:
                    image_names.append(filename)

        video_image_dict[video_name] = sorted(image_names)

    return video_image_dict

def predict_videos(predictor, res_folder, args):
    if osp.isdir(args.path):
        video_image_dict = get_video_image_dict(args.path)
    else:
        raise ValueError(f"args.path must be a directory, but got {args.path}")

    for video_name, files in video_image_dict.items():
        tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
        results = []
        for frame_id, img_name in enumerate(files, 1):
            video_image = f"{video_name}/{img_name}"
            outputs, img_info = predictor.inference(video_image)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1,1,1\n"
                        )

            # if frame_id % 20 == 0:
            #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        if args.save_result:
            res_file = osp.join(res_folder, f"{video_name}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_result:
        res_folder = osp.join(args.output_dir, "predictions", Path(args.path).stem)
        os.makedirs(res_folder, exist_ok=True)

    predictor = Predictor(args)
    predict_videos(predictor, res_folder, args)


if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--ann_path", type=str, default=None, help="path to coco annotation file")
    parser.add_argument("--pred_path", type=str, default=None, help="path to coco prediction file")
    args = parser.parse_args()
    main(args)
