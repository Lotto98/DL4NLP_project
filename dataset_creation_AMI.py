from datasets import load_dataset, Dataset
import os
import wget
from tqdm import tqdm
import json

import argparse
import pandas as pd

def create_annotations_ami(split:str="train"):
    
    def aggregate_continuous_segments(row:pd.Series, time_delta=0.1):
        
        #assuming that the time_pairs are sorted for each row
        
        new_speaker_ids = [row["speaker_id"][0]]
        new_time_pairs = [row["time_pairs"][0]]
        for speaker_id, time_pair in zip(row["speaker_id"], row["time_pairs"]):
            if speaker_id == new_speaker_ids[-1] and time_pair[0] < new_time_pairs[-1][1] + time_delta:
                max_end_time = max(new_time_pairs[-1][1], time_pair[1])
                new_time_pairs[-1] = (new_time_pairs[-1][0], max_end_time)            
            elif speaker_id == new_speaker_ids[-1] and (abs(time_pair[0] - new_time_pairs[-1][1]) <= time_delta):
                new_time_pairs[-1] = (new_time_pairs[-1][0], time_pair[1])
            else:
                new_speaker_ids.append(speaker_id)
                new_time_pairs.append(time_pair)
        
        new_row = row.copy()
        new_row['meeting_id'] = row['meeting_id']
        new_row["speaker_id"] = new_speaker_ids
        new_row["time_pairs"] = new_time_pairs
        
        return new_row
    
    def get_time_pairs_from_dataset(dataset:Dataset):
    
        # Convert the train dataset to a pandas DataFrame
        df = dataset.to_pandas()[["meeting_id","speaker_id","begin_time", "end_time"]]
        
        #print(sorted(df["speaker_id"].unique())) 

        df = df.sort_values(by=["meeting_id", "begin_time", "end_time"])        

        # Group by meeting_id and aggregate begin_time and end_time into lists
        aggregated_df = df.groupby('meeting_id', sort=False).agg(list).reset_index()
        #print(aggregated_df.head())

        # Merge the begin_time and end_time lists into one list of pairs using a vectorized operation
        aggregated_df['time_pairs'] = [list(zip(bt, et)) for bt, et in zip(aggregated_df['begin_time'], aggregated_df['end_time'])]
        
        # unify segments of the same speaker that are adjacent
        aggregated_df = aggregated_df.apply(aggregate_continuous_segments, axis=1)
        
        
        #[print(x[0], x[1], '\n') for x in zip(list(aggregated_df.loc[aggregated_df['meeting_id'] == 'ES2011a']['speaker_id']),
        #                                      list(aggregated_df.loc[aggregated_df['meeting_id'] == 'ES2011a']['time_pairs']))]
        
        aggregated_df = aggregated_df[['meeting_id',"time_pairs","speaker_id"]]
        
        return list(aggregated_df.to_dict("index").values()), aggregated_df["meeting_id"].unique()
    
    ds = load_dataset("edinburghcstr/ami", "ihm", split=split)
    ds_dict, meeting_ids = get_time_pairs_from_dataset(ds)
    
    """
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
    """
    
    os.makedirs("datasets/ami/annotations", exist_ok=True)
    
    with open(f'datasets/ami/annotations/{split}.json', 'w') as fp:
        json.dump(ds_dict, fp)
    
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