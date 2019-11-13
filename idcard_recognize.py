# -*- coding: utf-8 -*-
import idcardocr
import findidcard
import idcard_recognize


def process(img_name):
    try:
        idfind = findidcard.findidcard()
        idcard_img = idfind.find(img_name)
        result_dict = idcardocr.idcardocr(idcard_img)
        result_dict['error'] = 0
    except Exception as e:
        result_dict = {'error': 1}
        print(e)
    return result_dict


if __name__ == '__main__':
    print(idcard_recognize.process('testimages/3.jpg'))
