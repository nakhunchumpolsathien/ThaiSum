import pandas as pd
import os
from tqdm.notebook import tqdm

# read csv
df = pd.read_csv('thaisum.csv', encoding='utf-8')

'''
Conditions
________________________
if type is not null
________________________
1. 'ทั่วไทย'  = {'ภูมิภาค'} 
2. 'การเมือง' = {'ความมั่นคง', 'เลือกตั้ง',  }
3. 'สังคม' 
4. 'กีฬา' = {'ฟุตบอลยุโรป', 'ไทยรัฐเชียร์ไทยแลนด์', 'กีฬาอื่นๆ', 'ฟุตบอลไทย', 'มวย/MMA', 'ฟุตบอลโลก', 'วอลเลย์บอล', 'เอเชียนเกมส์', 'ไทยลีก', 'ฟุตซอล'}
5. 'ต่างประเทศ'
6. 'เศรษฐกิจ' = {ทองคำ}
7. 'ไลฟ์สไตล์' = {'ผู้หญิง', 'ท่องเที่ยว', 'อาหาร', 'ไลฟ์', 'บ้าน', 'หนัง'}
8. 'บันเทิง' = {'ศิลปะ-บันเทิง', 'วัฒนธรรม', 'ข่าวบันเทิง', ''}
9. 'คุณภาพชีวิต' = {'สิทธิมนุษยชน'}
10. 'วิทยาศาสตร์เทคโนโลยี'  = {'E-Sport', 'ไอซีที', 'วิทยาศาสตร์', 'การศึกษา'}
11. 'สิ่งแวดล้อม' = {'ภัยพิบัติ', ''}
12. 'unspecified' = { if row('tags') isnull and row('type') == 'unspecified'}
 
_________________________________
if row('type') == 'unspecified' 
_________________________________
1. 'ทั่วไทย'  = ['ข่าวทั่วไทย', ''ข่าวภูมิภาค', 'ทั่วไทย']
2. 'การเมือง' = ['ความมั่นคง', 'เลือกตั้ง',  'ข่าวการเมือง', 'คสช.', 'กกต', 'รัฐบาล', 'ยิ่งลักษณ์ ชินวัตร', 'การเลือกตั้ง', 'ร่างรัฐธรรมนูญ', 'ประชามติ', 'ประชาธิปัตย์', 'พรรคเพื่อไทย']
3. 'สังคม' = ['ข่าวสังคม', 'ข่าวโซเชียล']
4. 'กีฬา' = ['ข่าวกีฬา', 'พรีเมียร์ลีก', 'แมนเชสเตอร์ ยูไนเต็ด', 'ลิเวอร์พูล', 'ผลบอล', 'ทีมชาติไทย', 'เชลซี']
5. 'ต่างประเทศ' = ['ข่าวต่างประเทศ', 'จีน', 'สหรัฐ', 'อังกฤษ', 'ญี่ปุ่น']
6. 'เศรษฐกิจ' = ['ข่าวเศรษฐกิจ','ทองคำ', 'หวย', 'เศรษฐกิจ']
7. 'ไลฟ์สไตล์' = ['ผู้หญิง', 'ท่องเที่ยว', 'อาหาร', 'ไลฟ์', 'บ้าน', 'หนัง', 'ข่าวไลฟ์สไตล์']
8. 'บันเทิง' = ['ศิลปะ-บันเทิง', 'วัฒนธรรม', 'ข่าวบันเทิง', 'ดารา', 'ละคร', 'กอสซิป', 'นักร้อง',]
9. 'คุณภาพชีวิต' = ['สิทธิมนุษยชน', 'สุขภาพ', 'เกษตรกร', 'COVID19', 'ฆ่าตัวตาย', 'COVID-19', 'โควิด-19', 'ไวรัสโคโรนา', 'สาธารณสุข', ]
10. 'วิทยาศาสตร์เทคโนโลยี'  = {'E-Sport', 'ไอซีที', 'วิทยาศาสตร์', 'การศึกษา', 'เกษตร',  'ข่าวการศึกษา'}
11. 'สิ่งแวดล้อม' = ['ภัยพิบัติ', 'น้ำท่วม', 'ภัยแล้ง', 'กรมอุตุนิยมวิทยา', 'ไฟไหม้', 'พยากรณ์อากาศ', 'อากาศวันนี้', ]
'''

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    output_df = pd.DataFrame()
    if str(row['type']) == 'unspecified':  # if 'types' is NOT available ('unspecified'), then assign 'label' to the article from 'tags' as following conditions:

        if (pd.isnull(row['tags'])) and str(row['type']) == 'unspecified':
            label = 'unspecified'

        elif 'ความมั่นคง' in row["tags"] or 'เลือกตั้ง' in row["tags"] or 'ข่าวการเมือง' in row["tags"] or 'คสช.' in \
                row["tags"] or 'กกต' in row["tags"] or 'รัฐบาล' in row["tags"] or 'ยิ่งลักษณ์ ชินวัตร' in row[
            "tags"] or 'การเลือกตั้ง' in row["tags"] or 'ร่างรัฐธรรมนูญ' in row["tags"] or 'ประชามติ' in row[
            "tags"] or 'ประชาธิปัตย์' in row["tags"] or 'พรรคเพื่อไทย' in row["tags"] or 'การเมือง' in row["tags"]:
            label = 'politic'  # การเมือง

        elif 'ข่าวสังคม' in row["tags"] or 'ข่าวโซเชียล' in row["tags"]:
            label = 'society'  # สังคม

        elif 'ข่าวกีฬา' in row["tags"] or 'พรีเมียร์ลีก' in row["tags"] or 'แมนเชสเตอร์ ยูไนเต็ด' in row[
            "tags"] or 'ลิเวอร์พูล' in row["tags"] or 'ผลบอล' in row["tags"] or 'ทีมชาติไทย' in row[
            "tags"] or 'เชลซี' in row["tags"]:
            label = 'sport'  # กีฬา

        elif 'ข่าวต่างประเทศ' in row["tags"] or 'จีน' in row["tags"] or 'สหรัฐ' in row["tags"] or 'อังกฤษ' in row[
            "tags"] or 'ญี่ปุ่น' in row["tags"]:
            label = 'foreign'  # ต่างประเทศ

        elif 'ผู้หญิง' in row["tags"] or 'ท่องเที่ยว' in row["tags"] or 'อาหาร' in row["tags"] or 'ไลฟ์' in row[
            "tags"] or 'บ้าน' in row["tags"] or 'หนัง' in row["tags"] or 'ข่าวไลฟ์สไตล์' in row["tags"]:
            label = 'lifestyle'  # ไลฟ์สไตล์

        elif 'ข่าวเศรษฐกิจ' in row["tags"] or 'ทองคำ' in row["tags"] or 'หวย' in row["tags"] or 'เศรษฐกิจ' in row[
            "tags"]:
            label = 'economy'  # เศรษฐกิจ

        elif 'ศิลปะ-บันเทิง' in row["tags"] or 'วัฒนธรรม' in row["tags"] or 'ข่าวบันเทิง' in row["tags"] or 'ดารา' in \
                row["tags"] or 'ละคร' in row["tags"] or 'กอสซิป' in row["tags"] or 'นักร้อง' in row["tags"]:
            label = 'entertainment'  # บันเทิง & วัฒนธรรม

        elif 'สิทธิมนุษยชน' in row["tags"] or 'สุขภาพ' in row["tags"] or 'เกษตรกร' in row["tags"] or 'COVID19' in row[
            "tags"] or 'ฆ่าตัวตาย' in row["tags"] or 'COVID-19' in row["tags"] or 'โควิด-19' in row[
            "tags"] or 'ไวรัสโคโรนา' in row["tags"] or 'สาธารณสุข' in row["tags"]:
            label = 'quality-of-life'  # คุณภาพชีวิต

        elif 'E-Sport' in row["tags"] or 'ไอซีที' in row["tags"] or 'วิทยาศาสตร์' in row["tags"] or 'การศึกษา' in row[
            "tags"] or 'เกษตร' in row["tags"] or 'ข่าวการศึกษา' in row["tags"]:
            label = 'science'  # วิทยาศาสตร์ & การศึกษา

        elif 'ภัยพิบัติ' in row["tags"] or 'น้ำท่วม' in row["tags"] or 'ภัยแล้ง' in row["tags"] or 'กรมอุตุนิยมวิทยา' in \
                row["tags"] or 'ไฟไหม้' in row["tags"] or 'พยากรณ์อากาศ' in row["tags"] or 'อากาศวันนี้' in row["tags"]:
            label = 'environment'  # สิ่งแวดล้อม

        elif 'ข่าวทั่วไทย' in row["tags"] or 'ข่าวภูมิภาค' in row["tags"] or 'ทั่วไทย' in row["tags"]:
            label = 'local'  # ทั่วไทย

        else:
            label = 'others'  # อื่นๆ (This type of news doesn't have type but does have tags. However, these tags are not occoured many times.)

    else:  # if 'types' are already available, then assign 'label' to the article as following conditions:

        if 'ภัยพิบัติ' in row["type"] or 'สิ่งแวดล้อม' in row["type"]:
            label = 'environment'  # สิ่งแวดล้อม

        elif 'ความมั่นคง' in row["type"] or 'เลือกตั้ง' in row["type"] or 'การเมือง' in row["type"]:
            label = 'politic'  # การเมือง

        elif 'สังคม' in row["type"]:
            label = 'society'  # สังคม

        elif 'กีฬา' in row["type"] or 'ฟุตบอลยุโรป' in row["type"] or 'ไทยรัฐเชียร์ไทยแลนด์' in row[
            "type"] or 'กีฬาอื่นๆ' in row["type"] or 'ฟุตบอลไทย' in row["type"] or 'มวย/MMA' in row[
            "type"] or 'ฟุตบอลโลก' in row["type"] or 'วอลเลย์บอล' in row["type"] or 'เอเชียนเกมส์' in row[
            "type"] or 'ไทยลีก' in row["type"] or 'ฟุตซอล' in row["type"]:
            label = 'sport'  # กีฬา

        elif 'ต่างประเทศ' in row["type"]:
            label = 'foreign'  # ต่างประเทศ

        elif 'ผู้หญิง' in row["type"] or 'ท่องเที่ยว' in row["type"] or 'อาหาร' in row["type"] or 'ไลฟ์' in row[
            "type"] or 'บ้าน' in row["type"] or 'หนัง' in row["type"] or 'ไลฟ์สไตล์' in row["type"]:
            label = 'lifestyle'  # ไลฟ์สไตล์

        elif 'เศรษฐกิจ' in row["type"] or 'หวย' in row["type"] or 'ทองคำ' in row["type"]:
            label = 'economy'  # เศรษฐกิจ

        elif 'บันเทิง' in row["type"] or 'ศิลปะ-บันเทิง' in row["type"] or 'วัฒนธรรม' in row["type"] or 'ข่าวบันเทิง' in \
                row["type"]:
            label = 'entertainment'  # บันเทิง & วัฒนธรรม

        elif 'คุณภาพชีวิต' in row["type"] or 'สิทธิมนุษยชน' in row["type"]:
            label = 'quality-of-life'  # คุณภาพชีวิต

        elif 'วิทยาศาสตร์เทคโนโลยี' in row["type"] or 'E-Sport' in row["type"] or 'ไอซีที' in row[
            "type"] or 'วิทยาศาสตร์' in row["type"] or 'การศึกษา' in row["type"]:
            label = 'science'

        elif 'สิ่งแวดล้อม' in row["type"] or 'ภัยพิบัติ' in row["type"]:
            label = 'environment'  # สิ่งแวดล้อม

        elif 'ทั่วไทย' in row["type"] or 'ภูมิภาค' in row["type"]:
            label = 'local'  # ทั่วไทย

        else:
            label = 'others'  # อื่นๆ (This type of news does have tags. However, these types are not occoured many times.)

    output_df.loc[index, 'tags'] = str(row['tags'])
    output_df.loc[index, 'type'] = row['type']
    output_df.loc[index, 'label'] = label
    output_df.loc[index, 'url'] = row['url']

    output_name = os.path.join("tag-type-label-2.csv")
    if not os.path.isfile(output_name):
        output_df.to_csv(output_name, index=False, encoding='utf-8-sig', header=["tags", "type", "label", "url"])
    else:  # else it exists so append without writing the header
        output_df.to_csv(output_name, index=False, encoding='utf-8-sig', mode='a', header=False)
