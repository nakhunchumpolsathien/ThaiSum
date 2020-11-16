import glob
import os
import json
from tqdm.notebook import tqdm

json_path = "/content/drive/My Drive/Projects/ThaiSum-Dataset/simple-json/test-set-our-sent-segmentation/BertSum"
save_path = "/content/drive/My Drive/Projects/ThaiSum-Dataset/simple-json/test-set-our-sent-segmentation/ARedSum"

json_files = glob.glob(os.path.join(json_path, '*.json'))
id = 0

for json_file in tqdm(json_files):
    json_str = open(json_file, 'r', encoding='utf8').read()
    json_dict = json.loads(json_str)
    for i in tqdm(range(len(json_dict))):
        json_dict[i]['docId'] = id
        id += 1

    print('Writing json file')
    file_name = json_file.split('/')[-1]
    with open(os.path.join(save_path, file_name), 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False)

if __name__ == '__main__':
    pass
