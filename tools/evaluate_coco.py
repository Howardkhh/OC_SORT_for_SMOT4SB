import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def main():
    parser = argparse.ArgumentParser(description="Evaluate the object detection results.")
    parser.add_argument("--prediction", type=str, help="Path to the prediction JSON file.")
    parser.add_argument("--annotation", type=str, help="Path to the annotation JSON file.")
    args = parser.parse_args()

    coco_gt = COCO(args.annotation)
    coco_dt = coco_gt.loadRes(args.prediction)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()