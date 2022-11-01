
import argparse
import os
import shutil
from tqdm import tqdm

def main(inpath, outpath):
    if not os.path.exists(inpath): 
        print('Error: input path does not exist!!')
        return -1 
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
        os.makedirs(os.path.join(outpath, 'train'), exist_ok=True)
        os.makedirs(os.path.join(outpath, 'test'), exist_ok=True)

    for _f in tqdm(os.listdir(inpath)):
        parent_f = os.path.join(inpath, _f)
        if os.path.isdir(parent_f):
            for _ff in os.listdir(parent_f):
                parent_ff = os.path.join(parent_f, _ff)
                if os.path.isdir(parent_ff):
                    for ex in os.listdir(parent_ff):
                        seg_dir = os.path.join(parent_ff, ex)
                        if os.path.isdir(seg_dir):
                            for _file in os.listdir(seg_dir):
                                if _file.endswith('.PHN'):
                                    src_name = os.path.join(seg_dir, _file)
                                    tgt_name = os.path.join(outpath, _f.lower(), _ff+'_'+_file)
                                    tgt_name = os.path.splitext(tgt_name)[0]+'.phn'
                                    shutil.copy(src_name, tgt_name)
                                if _file.endswith('.WAV'): 
                                    src_name = os.path.join(seg_dir, _file)
                                    tgt_name = os.path.join(outpath, _f.lower(), _ff+'_'+_file)
                                    tgt_name = os.path.splitext(os.path.splitext(tgt_name)[0])[0]+'.wav'
                                    shutil.copy(src_name, tgt_name)

parser = argparse.ArgumentParser(description='Make TIMIT dataset ready phone segmentation')
parser.add_argument('--inpath', type=str, required=True, help='the path to the base timit dir.')
parser.add_argument('--outpath', type=str, required=True, help='the path to save the new format of the data.')

args = parser.parse_args()

main(args.inpath, args.outpath)

