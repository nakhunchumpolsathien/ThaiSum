import pandas as pd
from tqdm import tqdm

df = pd.read_csv('thaisum.csv', encoding='utf-8')

acr = atg = aya = bkk = bkn = brm = cbi = cco = cmi = cnt = cpm = cpn = cri = cti = kbi = kkn = kpt = kri = ksn = lei = lpg = lpn = lri = mdh = mkm = msn = nan = nbi = nbp = nki = nma = npm = npt = nrt = nsn = nwt = nyk = pbi = pct = pkn = pkt = plg = plk = pna = pnb = pre = pri = pte = ptn = pyo = rbr = ret = rng = ryg = sbr = ska = skm = skn = skw = sni = snk = spb = spk = sri = srn = ssk = sti = stn = tak = trg = trt = ubn = udn = utd = uti = yla = yst = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    if "กระบี่" in row["body"]:
        kbi = kbi + 1

    if "กรุงเทพมหานคร" in row["body"]:
        bkk = bkk + 1

    if "กาญจนบุรี" in row["body"]:
        kri = kri + 1

    if "กาฬสินธุ์" in row["body"]:
        ksn = ksn + 1

    if "กำแพงเพชร" in row["body"]:
        kpt = kpt + 1

    if "ขอนแก่น" in row["body"]:
        kkn = kkn + 1

    if "จันทบุรี" in row["body"]:
        cti = cti + 1

    if "ฉะเชิงเทรา" in row["body"]:
        cco = cco + 1

    if "ชลบุรี" in row["body"]:
        cbi = cbi + 1

    if "ชัยนาท" in row["body"]:
        cnt = cnt + 1

    if "ชัยภูมิ" in row["body"]:
        cpm = cpm + 1

    if "ชุมพร" in row["body"]:
        cpn = cpn + 1

    if "เชียงราย" in row["body"]:
        cri = cri + 1

    if "เชียงใหม่" in row["body"]:
        cmi = cmi + 1

    if "ตรัง" in row["body"]:
        trg = trg + 1

    if "ตราด" in row["body"]:
        trt = trt + 1

    if "ตาก" in row["body"]:
        tak = tak + 1

    if "นครนายก" in row["body"]:
        nyk = nyk + 1

    if "นครปฐม" in row["body"]:
        npt = npt + 1

    if "นครพนม" in row["body"]:
        npm = npm + 1

    if "นครราชสีมา" in row["body"]:
        nma = nma + 1

    if "นครศรีธรรมราช" in row["body"]:
        nrt = nrt + 1

    if "นครสวรรค์" in row["body"]:
        nsn = nsn + 1

    if "นนทบุรี" in row["body"]:
        nbi = nbi + 1

    if "นราธิวาส" in row["body"]:
        nwt = nwt + 1

    if "น่าน" in row["body"]:
        nan = nan + 1

    if "บึงกาฬ" in row["body"]:
        bkn = bkn + 1

    if "บุรีรัมย์" in row["body"]:
        brm = brm + 1

    if "ปทุมธานี" in row["body"]:
        pte = pte + 1

    if "ประจวบคีรีขันธ์" in row["body"]:
        pkn = pkn + 1

    if "ปราจีนบุรี" in row["body"]:
        pri = pri + 1

    if "ปัตตานี" in row["body"]:
        ptn = ptn + 1

    if "พะเยา" in row["body"]:
        pyo = pyo + 1

    if "พระนครศรีอยุธยา" in row["body"]:
        aya = aya + 1

    if "พังงา" in row["body"]:
        pna = pna + 1

    if "พัทลุง" in row["body"]:
        plg = plg + 1

    if "พิจิตร" in row["body"]:
        pct = pct + 1

    if "พิษณุโลก" in row["body"]:
        plk = plk + 1

    if "เพชรบุรี" in row["body"]:
        pbi = pbi + 1

    if "เพชรบูรณ์" in row["body"]:
        pnb = pnb + 1

    if "แพร่" in row["body"]:
        pre = pre + 1

    if "ภูเก็ต" in row["body"]:
        pkt = pkt + 1

    if "มหาสารคาม" in row["body"]:
        mkm = mkm + 1

    if "มุกดาหาร" in row["body"]:
        mdh = mdh + 1

    if "แม่ฮ่องสอน" in row["body"]:
        msn = msn + 1

    if "ยโสธร" in row["body"]:
        yst = yst + 1

    if "ยะลา" in row["body"]:
        yla = yla + 1

    if "ร้อยเอ็ด" in row["body"]:
        ret = ret + 1

    if "ระนอง" in row["body"]:
        rng = rng + 1

    if "ระยอง" in row["body"]:
        ryg = ryg + 1

    if "ราชบุรี" in row["body"]:
        rbr = rbr + 1

    if "ลพบุรี" in row["body"]:
        lri = lri + 1

    if "ลำปาง" in row["body"]:
        lpg = lpg + 1

    if "ลำพูน" in row["body"]:
        lpn = lpn + 1

    if "เลย" in row["body"]:
        lei = lei + 1

    if "ศรีสะเกษ" in row["body"]:
        ssk = ssk + 1

    if "สกลนคร" in row["body"]:
        snk = snk + 1

    if "สงขลา" in row["body"]:
        ska = ska + 1

    if "สตูล" in row["body"]:
        stn = stn + 1

    if "สมุทรสงคราม" in row["body"]:
        skm = skm + 1

    if "สมุทรปราการ" in row["body"]:
        spk = spk + 1

    if "สมุทรสาคร" in row["body"]:
        skn = skn + 1

    if "สระแก้ว" in row["body"]:
        skw = skw + 1

    if "สระบุรี" in row["body"]:
        sri = sri + 1

    if "สิงห์บุรี" in row["body"]:
        sbr = sbr + 1

    if "สุโขทัย" in row["body"]:
        sti = sti + 1

    if "สุพรรณบุรี" in row["body"]:
        spb = spb + 1

    if "สุราษฎร์ธานี" in row["body"]:
        sni = sni + 1

    if "สุรินทร์" in row["body"]:
        srn = srn + 1

    if "หนองคาย" in row["body"]:
        nki = nki + 1

    if "หนองบัวลำภู" in row["body"]:
        nbp = nbp + 1

    if "อ่างทอง" in row["body"]:
        atg = atg + 1

    if "อำนาจเจริญ" in row["body"]:
        acr = acr + 1

    if "อุดรธานี" in row["body"]:
        udn = udn + 1

    if "อุตรดิตถ์" in row["body"]:
        utd = utd + 1

    if "อุทัยธานี" in row["body"]:
        uti = uti + 1

    if "อุบลราชธานี" in row["body"]:
        ubn = ubn + 1
