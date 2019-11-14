# -*- coding: utf-8 -*-
import re

from PIL import Image
import cv2
import numpy as np
import pytesseract

# windows下需要修改pytesseract中的地址（不一定）
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
x = 1280.00 / 3840.00
pixel_x = int(x * 3840)


# print(x, pixel_x)

# mode1：识别所有信息
def idcardocr(imgname, mode=1):
    print(u'进入身份证识别流程...')
    if mode == 1:
        # generate_mask(x)
        img_data_gray, img_org = img_resize_gray(imgname)
        result_dict = dict()
        name_pic = find_name(img_data_gray, img_org)
        # showimg(name_pic)
        # print 'name'
        result_dict['name'] = get_name(name_pic)
        # print result_dict['name']

        sex_pic = find_sex(img_data_gray, img_org)
        # showimg(sex_pic)
        # print 'sex'
        result_dict['sex'] = get_sex(sex_pic)
        # print result_dict['sex']

        nation_pic = find_nation(img_data_gray, img_org)
        # showimg(nation_pic)
        # print 'nation'
        result_dict['nation'] = get_nation(nation_pic)
        # print result_dict['nation']

        address_pic = find_address(img_data_gray, img_org)
        # showimg(address_pic)
        # print 'address'
        result_dict['address'] = get_address(address_pic)
        # print result_dict['address']

        idnum_pic = find_idnum(img_data_gray, img_org)
        # showimg(idnum_pic)
        # print 'idnum'
        result_dict['idnum'], result_dict['birth'] = get_idnum_and_birth(idnum_pic)
        # print result_dict['idnum']
    else:
        print(u"模式选择错误！")
    # showimg(img_data_gray)
    return result_dict


# idcardocr里面resize以高度为依据, 用于get部分
def img_resize(imggray, dheight):
    # print 'dheight:%s' % dheight
    crop = imggray
    size = crop.get().shape
    height = size[0]
    width = size[1]
    width = width * dheight / height
    crop = cv2.resize(src=crop, dsize=(int(width), dheight), interpolation=cv2.INTER_CUBIC)
    return crop


def img_resize_gray(imgorg):
    # imgorg = cv2.imread(imgname)
    crop = imgorg
    size = cv2.UMat.get(crop).shape
    # print size
    height = size[0]
    width = size[1]
    # 参数是根据3840调的
    height = int(height * 3840 * x / width)
    # print height
    crop = cv2.resize(src=crop, dsize=(int(3840 * x), height), interpolation=cv2.INTER_CUBIC)
    return hist_equal(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), crop


def find_name(crop_gray, crop_org):
    template = cv2.UMat(cv2.imread('./mask/name_mask_%s.jpg' % pixel_x, 0))
    # showimg(crop_org)
    w, h = cv2.UMat.get(template).shape[::-1]
    res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # print(max_loc)
    top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(700 * x), top_left[1] + int(300 * x))
    result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
    cv2.rectangle(crop_gray, top_left, bottom_right, 255, 2)
    # showimg(result)
    return cv2.UMat(result)


def find_sex(crop_gray, crop_org):
    template = cv2.UMat(cv2.imread('./mask/sex_mask_%s.jpg' % pixel_x, 0))
    # showimg(template)
    w, h = cv2.UMat.get(template).shape[::-1]
    res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(300 * x), top_left[1] + int(300 * x))
    result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
    cv2.rectangle(crop_gray, top_left, bottom_right, 255, 2)
    # showimg(crop_gray)
    return cv2.UMat(result)


def find_nation(crop_gray, crop_org):
    template = cv2.UMat(cv2.imread('./mask/nation_mask_%s.jpg' % pixel_x, 0))
    # showimg(template)
    w, h = cv2.UMat.get(template).shape[::-1]
    res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = (max_loc[0] + w - int(20 * x), max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(500 * x), top_left[1] + int(300 * x))
    result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
    cv2.rectangle(crop_gray, top_left, bottom_right, 255, 2)
    # showimg(crop_gray)
    return cv2.UMat(result)


def find_address(crop_gray, crop_org):
    template = cv2.UMat(cv2.imread('./mask/address_mask_%s.jpg' % pixel_x, 0))
    # showimg(template)
    # showimg(crop_gray)
    w, h = cv2.UMat.get(template).shape[::-1]
    # t1 = round(time.time()*1000)
    res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    # t2 = round(time.time()*1000)
    # print 'time:%s'%(t2-t1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(1700 * x), top_left[1] + int(550 * x))
    result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
    cv2.rectangle(crop_gray, top_left, bottom_right, 255, 2)
    # showimg(crop_gray)
    return cv2.UMat(result)


def find_idnum(crop_gray, crop_org):
    template = cv2.UMat(cv2.imread('./mask/idnum_mask_%s.jpg' % pixel_x, 0))
    # showimg(template)
    # showimg(crop_gray)
    w, h = cv2.UMat.get(template).shape[::-1]
    res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(2300 * x), top_left[1] + int(300 * x))
    result = cv2.UMat.get(crop_org)[top_left[1] - 10:bottom_right[1], top_left[0] - 10:bottom_right[0]]
    cv2.rectangle(crop_gray, top_left, bottom_right, 255, 2)
    # showimg(crop_gray)
    return cv2.UMat(result)


def showimg(img):
    cv2.namedWindow("contours", 0);
    cv2.resizeWindow("contours", 1280, 720);
    cv2.imshow("contours", img)
    cv2.waitKey()


# psm model:
#  0    Orientation and script detection (OSD) only.
#  1    Automatic page segmentation with OSD.
#  2    Automatic page segmentation, but no OSD, or OCR.
#  3    Fully automatic page segmentation, but no OSD. (Default)
#  4    Assume a single column of text of variable sizes.
#  5    Assume a single uniform block of vertically aligned text.
#  6    Assume a single uniform block of text.
#  7    Treat the image as a single text line.
#  8    Treat the image as a single word.
#  9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
# 			bypassing hacks that are Tesseract-specific

def get_name(img):
    #    cv2.imshow("method3", img)
    #    cv2.waitKey()
    # print('name')
    _, _, red = cv2.split(img)  # split 会自动将UMat转换回Mat
    red = cv2.UMat(red)
    red = hist_equal(red)
    red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 50)
    #    red = cv2.medianBlur(red, 3)
    red = img_resize(red, 150)
    img = img_resize(img, 150)
    # showimg(red)
    # cv2.imwrite('name.png', red)
    #    img2 = Image.open('address.png')
    # img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
    # return get_result_vary_length(red, 'chi_sim', img, '-psm 7')
    return get_result_vary_length(red, 'chi_sim', img, '--psm 7')
    # return punc_filter(pytesseract.image_to_string(img, lang='chi_sim', config='-psm 13').replace(" ",""))


def get_sex(img):
    _, _, red = cv2.split(img)
    # print('sex')
    red = cv2.UMat(red)
    red = hist_equal(red)
    red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    #    red = cv2.medianBlur(red, 3)
    #    cv2.imwrite('address.png', img)
    #    img2 = Image.open('address.png')
    red = img_resize(red, 150)
    # cv2.imwrite('sex.png', red)
    # img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
    return get_result_fix_length(red, 1, 'sex', '--psm 10')
    # return get_result_fix_length(red, 1, 'chi_sim', '--psm 10')
    # return pytesseract.image_to_string(img, lang='sex', config='-psm 10').replace(" ","")


def get_nation(img):
    _, _, red = cv2.split(img)
    # print('nation')
    red = cv2.UMat(red)
    red = hist_equal(red)
    red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    red = img_resize(red, 150)
    # cv2.imwrite('nation.png', red)
    # img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
    return get_result_fix_length(red, 1, 'nation', '--psm 7')
    # return get_result_fix_length(red, 1, 'chi_sim', '--psm 10')
    # return pytesseract.image_to_string(img, lang='nation', config='-psm 13').replace(" ","")


def get_address(img):
    # _, _, red = cv2.split(img)
    # red = cv2.medianBlur(red, 3)
    # print('address')
    _, _, red = cv2.split(img)
    red = cv2.UMat(red)
    red = hist_equal(red)
    red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    red = img_resize(red, 300)
    # img = img_resize(img, 300)
    # cv2.imwrite('address_red.png', red)
    img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
    # return punc_filter(get_result_vary_length(red,'chi_sim', img, '-psm 6'))
    return punc_filter(get_result_vary_length(red, 'chi_sim', img, '--psm 6'))
    # return punc_filter(pytesseract.image_to_string(img, lang='chi_sim', config='-psm 3').replace(" ",""))


def get_idnum_and_birth(img):
    _, _, red = cv2.split(img)
    # print('idnum')
    red = cv2.UMat(red)
    red = hist_equal(red)
    red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    red = img_resize(red, 150)
    # cv2.imwrite('idnum_red.png', red)
    # idnum_str = get_result_fix_length(red, 18, 'idnum', '-psm 8')
    # idnum_str = get_result_fix_length(red, 18, 'eng', '--psm 8 ')
    img = Image.fromarray(cv2.UMat.get(red).astype('uint8'))
    idnum_str = get_result_vary_length(red, 'eng', img, '--psm 8 ')
    return idnum_str, idnum_str[6:14]


def get_result_fix_length(red, fix_length, langset, custom_config=''):
    red_org = red
    cv2.fastNlMeansDenoising(red, red, 4, 7, 35)
    rec, red = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # 描边一次可以减少噪点
    cv2.drawContours(red, contours, -1, (0, 255, 0), 1)

    h_threshold = 54
    numset_contours = []
    calcu_cnt = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > h_threshold:
            numset_contours.append((x, y, w, h))
    while len(numset_contours) != fix_length:
        if calcu_cnt > 50:
            print(u'计算次数过多！目前阈值为：', h_threshold)
            break
        numset_contours = []
        calcu_cnt += 1
        if len(numset_contours) > fix_length:
            h_threshold += 1
            contours_cnt = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > h_threshold:
                    contours_cnt += 1
                    numset_contours.append((x, y, w, h))
        if len(numset_contours) < fix_length:
            h_threshold -= 1
            contours_cnt = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > h_threshold:
                    contours_cnt += 1
                    numset_contours.append((x, y, w, h))
    result_string = ''
    numset_contours.sort(key=lambda num: num[0])
    for x, y, w, h in numset_contours:
        result_string += pytesseract.image_to_string(cv2.UMat.get(red_org)[y - 10:y + h + 10, x - 10:x + w + 10],
                                                     lang=langset, config=custom_config)
    # print(new_r)
    # cv2.imwrite('fixlengthred.png', cv2.UMat.get(red_org)[y-10:y + h +10 , x-10:x + w + 10])
    # print(result_string)
    return result_string


def get_result_vary_length(red, langset, org_img, custom_config=''):
    red_org = red
    # cv2.fastNlMeansDenoising(red, red, 4, 7, 35)
    rec, red = cv2.threshold(red, 127, 255, cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # 描边一次可以减少噪点
    cv2.drawContours(red, contours, -1, (255, 255, 255), 1)
    color_img = cv2.cvtColor(red, cv2.COLOR_GRAY2BGR)
    numset_contours = []
    height_list = []
    width_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height_list.append(h)
        # print(h,w)
        width_list.append(w)
    height_list.remove(max(height_list))
    width_list.remove(max(width_list))
    height_threshold = 0.70 * max(height_list)
    width_threshold = 1.4 * max(width_list)
    # print('height_threshold:'+str(height_threshold)+'width_threshold:'+str(width_threshold))
    big_rect = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > height_threshold and w < width_threshold:
            # print(h,w)
            numset_contours.append((x, y, w, h))
            big_rect.append((x, y))
            big_rect.append((x + w, y + h))
    big_rect_nparray = np.array(big_rect, ndmin=3)
    x, y, w, h = cv2.boundingRect(big_rect_nparray)
    # imgrect = cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # showimg(imgrect)
    # showimg(cv2.UMat.get(org_img)[y:y + h, x:x + w])

    result_string = ''
    result_string += pytesseract.image_to_string(cv2.UMat.get(red_org)[y - 10:y + h + 10, x - 10:x + w + 10],
                                                 lang=langset,
                                                 config=custom_config)
    # print(result_string)
    # cv2.imwrite('varylength.png', cv2.UMat.get(org_img)[y:y + h, x:x + w])
    # cv2.imwrite('varylengthred.png', cv2.UMat.get(red_org)[y:y + h, x:x + w])
    # numset_contours.sort(key=lambda num: num[0])
    # for x, y, w, h in numset_contours:
    #     result_string += pytesseract.image_to_string(cv2.UMat.get(color_img)[y:y + h, x:x + w], lang=langset, config=custom_config)
    return punc_filter(result_string)


def punc_filter(str):
    temp = str
    xx = u"([\u4e00-\u9fff0-9A-Z]+)"
    pattern = re.compile(xx)
    results = pattern.findall(temp)
    string = ""
    for result in results:
        string += result
    return string


# 这里使用直方图拉伸，不是直方图均衡
def hist_equal(img):
    # clahe_size = 8
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(clahe_size, clahe_size))
    # result = clahe.apply(img)
    # test
    # result = cv2.equalizeHist(img)
    image = img.get()  # UMat to Mat
    # result = cv2.equalizeHist(image)
    lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表
    # lut = np.zeros(256)
    hist = cv2.calcHist([image],  # 计算图像的直方图
                        [0],  # 使用的通道
                        None,  # 没有使用mask
                        [256],  # it is a 1D histogram
                        [0, 256])
    minBinNo, maxBinNo = 0, 255
    # 计算从左起第一个不为0的直方图柱的位置
    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break
    # 计算从右起第一个不为0的直方图柱的位置
    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255 - binNo
            break
    # print minBinNo, maxBinNo
    # 生成查找表
    for i, v in enumerate(lut):
        if i < minBinNo:
            lut[i] = 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            lut[i] = int(255.0 * (i - minBinNo) / (maxBinNo - minBinNo) + 0.5)
    # 计算,调用OpenCV cv2.LUT函数,参数 image --  输入图像，lut -- 查找表
    # print lut
    result = cv2.LUT(image, lut)
    # print type(result)
    # showimg(result)
    return cv2.UMat(result)
