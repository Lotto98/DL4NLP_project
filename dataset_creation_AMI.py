from datasets import load_dataset, Dataset
import os
import wget
from tqdm import tqdm
import json

import argparse

def create_annotations_ami(split:str="train"):
    
    def get_time_pairs_from_dataset(dataset:Dataset):
    
        # Convert the train dataset to a pandas DataFrame
        df = dataset.to_pandas()[["meeting_id","speaker_id","begin_time", "end_time"]]

        # Group by meeting_id and aggregate begin_time and end_time into lists
        aggregated_df = df.groupby('meeting_id').agg(list).reset_index()

        # Merge the begin_time and end_time lists into one list of pairs using a vectorized operation
        aggregated_df['time_pairs'] = [list(zip(bt, et)) for bt, et in zip(aggregated_df['begin_time'], aggregated_df['end_time'])]

        aggregated_df = aggregated_df[['meeting_id',"time_pairs","speaker_id"]]
        #aggregated_df['time_pairs'] = aggregated_df['time_pairs'].apply(func=sorted)
        
        return list(aggregated_df.to_dict("index").values()), aggregated_df["meeting_id"].unique()
    
    ds = load_dataset("edinburghcstr/ami", "ihm", split=split)
    ds_dict, meeting_ids = get_time_pairs_from_dataset(ds)
    
    annotations = []
    
    for meeting in ds_dict:
        meeting_id = meeting["meeting_id"]
        time_pairs = meeting["time_pairs"]
        speaker_ids = meeting["speaker_id"]
        for time_pair, speaker_id in zip(time_pairs, speaker_ids):
            annotations.append({
                "file_name": meeting_id,
                "time_pair": time_pair,
                "speaker_id": speaker_id
            })

    os.makedirs("datasets/ami/annotations", exist_ok=True)
    
    with open(f'datasets/ami/annotations/{split}.json', 'w') as fp:
        json.dump(annotations, fp)
    
    return meeting_ids

def create_audio_ami(meeting_ids: list, split:str="train"):
    
    os.makedirs("datasets/ami/audio", exist_ok=True)
    
    def update_bar(current, total, width):
        download_bar.total = total
        download_bar.update(current - download_bar.n)
        
    
    for meeting_id in tqdm(meeting_ids, desc=f"Downloading {split} audio files"):
        
        download_bar = tqdm(total=1, position=1, leave=False, desc=f"Downloading {meeting_id}.wav")
        
        if os.path.exists(f"datasets/ami/audio/{meeting_id}.wav"):
            continue
        
        url = f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
        wget.download(url, out=f"datasets/ami/audio/{meeting_id}.wav", bar=update_bar)

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", type=str, default="all", help="Split to create the dataset for. Can be 'train', 'validation', 'test' or 'all'")
    args = argparser.parse_args()
    
    if args.split == "all":
        meeting_ids = create_annotations_ami("train")
        create_audio_ami(meeting_ids, "train")
        
        meeting_ids = create_annotations_ami("validation")
        create_audio_ami(meeting_ids, "validation")
        
        meeting_ids = create_annotations_ami("test")
        create_audio_ami(meeting_ids, "test")
    else:
        meeting_ids = create_annotations_ami(args.split)
        create_audio_ami(meeting_ids, args.split)