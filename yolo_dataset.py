from diffusiondet.dataset_audio import DiffusionDetAudioDataset
from argparse import Namespace, ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test", "all"])
    args = parser.parse_args()
    
    
    #create a yaml file for the dataset
    yaml_file = f"datasets/ami_yolo/ami.yaml"
    
    content =f"""
path: dataset_yolo/ami # dataset root dir
train: images/train 
val: images/val 
test: images/test

names:
    0: man
    1: female
"""
    
    with open(yaml_file, "w") as f:
        f.writelines(content)
    
    
    INPUT = Namespace(SAMPLING_RATE=16000, SECONDS_PER_SEGMENT=11)
    cfg = Namespace(INPUT=INPUT)

    if args.split == "all":
        DiffusionDetAudioDataset( cfg=cfg, name="ami", split="train").to_yolo_format()
        DiffusionDetAudioDataset( cfg=cfg, name="ami", split="validation").to_yolo_format()
        DiffusionDetAudioDataset( cfg=cfg, name="ami", split="test").to_yolo_format()
    else:
        DiffusionDetAudioDataset( cfg=cfg, name="ami", split=args.split).to_yolo_format()