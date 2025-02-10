import shutil
from datasets import load_dataset, Dataset
import os
import wget
from tqdm import tqdm
import json

import argparse
import pandas as pd
import xml.etree.ElementTree as ET

def create_annotations_ami_from_hugg(split:str="train"):
    
    def aggregate_continuous_segments(row:pd.Series, time_delta=0.1):
        
        #assuming that the time_pairs are sorted for each row
        
        new_speaker_ids = [row["speaker_id"][0]]
        new_time_pairs = [row["time_pairs"][0]]
        new_text = [row["text"][0]]
        for speaker_id, time_pair, text in zip(row["speaker_id"], row["time_pairs"], row["text"]):
            if speaker_id == new_speaker_ids[-1] and time_pair[0] < new_time_pairs[-1][1] + time_delta:
                max_end_time = max(new_time_pairs[-1][1], time_pair[1])
                new_time_pairs[-1] = (new_time_pairs[-1][0], max_end_time)
                
                if max_end_time == time_pair[1]:
                    new_text[-1] += " " + text
                else:
                    new_text[-1] = text + " " + new_text[-1]
        
            elif speaker_id == new_speaker_ids[-1] and (abs(time_pair[0] - new_time_pairs[-1][1]) <= time_delta):
                new_time_pairs[-1] = (new_time_pairs[-1][0], time_pair[1])
                new_text[-1] = text + " " + new_text[-1]
            else:
                new_speaker_ids.append(speaker_id)
                new_time_pairs.append(time_pair)
                new_text.append(text)
        
        new_row = row.copy()
        new_row['meeting_id'] = row['meeting_id']
        new_row["speaker_id"] = new_speaker_ids
        new_row["time_pairs"] = new_time_pairs
        
        #print(new_row)
        
        return new_row
    
    def get_time_pairs_from_dataset(dataset:Dataset):
    
        # Convert the train dataset to a pandas DataFrame
        df = dataset.to_pandas()[["meeting_id","speaker_id","begin_time", "end_time", "text"]]
        
        #print(df[df["speaker_id"] == "MEE071"])
        
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
        
        aggregated_df = aggregated_df[['meeting_id',"time_pairs","speaker_id", "text"]]
        
        return list(aggregated_df.to_dict("index").values()), aggregated_df["meeting_id"].unique()
    
    ds = load_dataset("edinburghcstr/ami", "ihm", split=split, trust_remote_code=True)
    ds_dict, meeting_ids = get_time_pairs_from_dataset(ds)
    
    os.makedirs("datasets/ami/annotations", exist_ok=True)
    
    speaker_ids = set()
    for meeting in ds_dict:
        speaker_ids.update(meeting["speaker_id"])
    
    with open(f'datasets/ami/annotations/{split}.json', 'w') as fp:
        json.dump(ds_dict, fp)
    
    return meeting_ids, speaker_ids

def create_annotations_ami_from_file(split:str="train"):
    
    def get_meeting_ids(split:str):
        
        if split=="train":
            ids = ['EN2001a', 'EN2001b', 'EN2001d', 'EN2001e', 'EN2003a' ,'EN2004a' 'EN2005a' ,
                    'EN2006a', 'EN2006b', 'EN2009b', 'EN2009c', 'EN2009d', 'ES2002a' 'ES2002b',
                    'ES2002c', 'ES2002d', 'ES2003a', 'ES2003b', 'ES2003c', 'ES2003d' 'ES2005a',
                    'ES2005b', 'ES2005c', 'ES2005d', 'ES2006a', 'ES2006b', 'ES2006c' 'ES2006d',
                    'ES2007a', 'ES2007b', 'ES2007c', 'ES2007d', 'ES2008a', 'ES2008b' 'ES2008c',
                    'ES2008d', 'ES2009a', 'ES2009b', 'ES2009c', 'ES2009d', 'ES2010a' 'ES2010b',
                    'ES2010c', 'ES2010d', 'ES2012a', 'ES2012b', 'ES2012c', 'ES2012d' 'ES2013a',
                    'ES2013b', 'ES2013c', 'ES2013d', 'ES2014a', 'ES2014b', 'ES2014c' 'ES2014d',
                    'ES2015a', 'ES2015b', 'ES2015c', 'ES2015d', 'ES2016a', 'ES2016b' 'ES2016c',
                    'ES2016d', 'IB4005', 'IN1001', 'IN1002', 'IN1005', 'IN1007', 'IN1008', 'IN1009',
                    'IN1012', 'IN1013', 'IN1014', 'IN1016', 'IS1000a', 'IS1000b', 'IS1000c',
                    'IS1000d', 'IS1001a', 'IS1001b', 'IS1001c', 'IS1001d' ,'IS1002b' ,'IS1002c',
                    'IS1002d', 'IS1003a', 'IS1003b', 'IS1003c', 'IS1003d', 'IS1004a', 'IS1004b',
                    'IS1004c', 'IS1004d', 'IS1005a', 'IS1005b', 'IS1005c', 'IS1006a', 'IS1006b',
                    'IS1006c', 'IS1006d', 'IS1007a', 'IS1007b', 'IS1007c', 'IS1007d', 'TS3005a',
                    'TS3005b', 'TS3005c', 'TS3005d', 'TS3006a', 'TS3006b', 'TS3006c', 'TS3006d',
                    'TS3007a', 'TS3007b', 'TS3007c', 'TS3007d', 'TS3008a', 'TS3008b', 'TS3008c',
                    'TS3008d', 'TS3009a', 'TS3009b', 'TS3009c', 'TS3009d', 'TS3010a', 'TS3010b',
                    'TS3010c', 'TS3010d', 'TS3011a', 'TS3011b', 'TS3011c', 'TS3011d', 'TS3012a',
                    'TS3012b', 'TS3012c', 'TS3012d']
        elif split == "validation":
            ids = ['ES2011a', 'ES2011b', 'ES2011c', 'ES2011d', 'IB4001', 'IB4002', 'IB4003'
                    'IB4004', 'IB4010', 'IB4011', 'IS1008a', 'IS1008b', 'IS1008c', 'IS1008d'
                    'TS3004a', 'TS3004b', 'TS3004c', 'TS3004d']
        elif split == "test":
            ids = ['EN2002a', 'EN2002b', 'EN2002c', 'EN2002d', 'ES2004a', 'ES2004b', 'ES2004c',
                    'ES2004d', 'IS1009a', 'IS1009b', 'IS1009c', 'IS1009d', 'TS3003a', 'TS3003b',
                    'TS3003c', 'TS3003d']
        
        return ids
    
    def download_annotations(split:str="train"):
        
        url = f"https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
        wget.download(url, out=f"temp.zip")
        
        folder_path = "temp"
        os.makedirs(folder_path, exist_ok=True)
        os.system(f"unzip temp.zip -d temp")
        
        folder_path = os.path.join(folder_path, "segments")
        
        # List all XML files in the folder
        xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

        # Sort the files if needed (e.g., by filename)
        xml_files.sort()

        annotations = {} # meeting_id -> time_pairs
        
        # Load and parse each XML file
        for xml_file in xml_files:
            xml_path = os.path.join(folder_path, xml_file)
            
            meeting_id = xml_file.split(".")[0]
            
            print(meeting_id)
            
            if meeting_id not in get_meeting_ids(split):
                continue
            
            #print("Meeting ID:", meeting_id)
            
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            #print("File:", xml_file)
            
            for child in root:
                segment = child.attrib
                start, end = float(segment["transcriber_start"]), float(segment["transcriber_end"])
                if meeting_id not in annotations:
                    annotations[meeting_id] = []
                annotations[meeting_id].append([start, end])
        
        #sort the time pairs
        for meeting_id in annotations:
            annotations[meeting_id].sort()
        
        shutil.rmtree('temp')
        os.remove('temp.zip')
        
        return annotations
    
    def create_json_file(annotations:dict, split:str):
        
        new_annotations = []
        
        for meeting_id, time_pairs in annotations.items():
            new_annotations.append({"meeting_id": meeting_id, "time_pairs": time_pairs, "speaker_id": ["M"]*len(time_pairs), "text": [""]*len(time_pairs)})
        
        os.makedirs("datasets/ami/annotations", exist_ok=True)
        
        with open(f'datasets/ami/annotations/{split}.json', 'w') as fp:
            json.dump(new_annotations, fp)
        
    annotations = download_annotations(split)
    create_json_file(annotations, split)
    
    return list(annotations.keys()), [0]

def create_annotations_ami(split:str="train", from_hugg=False):
    
    if from_hugg:
        pass
        #return create_annotations_ami_from_hugg(split)
    else:
        return create_annotations_ami_from_file(split)

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
        meeting_ids, unique_speakers_ids_train = create_annotations_ami("train")
        print(meeting_ids)
        create_audio_ami(meeting_ids, "train")
        
        meeting_ids, unique_speakers_ids_validation = create_annotations_ami("validation")
        print(meeting_ids)
        create_audio_ami(meeting_ids, "validation")
        
        meeting_ids, unique_speakers_ids_test = create_annotations_ami("test")
        print(meeting_ids)
        create_audio_ami(meeting_ids, "test")
        
        #unique_ids = list(unique_speakers_ids_train | unique_speakers_ids_test)
        
        print()
        
        print(len(unique_speakers_ids_train))
        print(len(unique_speakers_ids_validation))
        print(len(unique_speakers_ids_test))
        
        #print(len(unique_ids))
        
        #encoder = LabelEncoder().fit(unique_ids)
        
        #with open("datasets/ami/classes.pkl", 'wb') as f:
        #    pickle.dump(encoder, f)
    else:
        meeting_ids, unique_speakers_ids = create_annotations_ami_from_file(args.split)
        print(meeting_ids)
        create_audio_ami(meeting_ids, args.split)