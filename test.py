# -*- coding: utf-8 -*-
# import idcardocr
import findidcard
import recognizeidcard


def process(img_name):
    try:
        idfind = findidcard.findidcard()
        idcard_img = idfind.find(img_name)
        result_dict = recognizeidcard.idcardocr(idcard_img, 1)
        result_dict['error'] = 0
    except Exception as e:
        result_dict = {'error': 1}
        print(e)
    return result_dict


if __name__ == '__main__':
    idcardimagepath = 'testimages/11.jpg'
    info = process(idcardimagepath)
    error = info['error']
    if error == 0:
        name = info['name']
        nation = info['nation']
        sex = info['sex']
        birth = info['birth']
        address = info['address']
        idnum = info['idnum']
        print('name:   ' + name)
        print('nation: ' + nation)
        print('sex:    ' + sex)
        print('birth:  ' + birth)
        print('address:' + address)
        print('idnum:  ' + idnum)
    else:
        print(info)
