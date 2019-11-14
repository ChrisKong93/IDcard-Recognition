import idcard_recognize

if __name__ == '__main__':
    idcardimagepath = 'testimages/11.jpg'
    info = idcard_recognize.process(idcardimagepath)
    error = info['error']
    if error == 0:
        name = info['name']
        nation = info['nation']
        sex = info['sex']
        birth = info['birth']
        address = info['address']
        idnum = info['idnum']
        print('*'*30)
        print('name:   ' + name)
        print('nation: ' + nation)
        print('sex:    ' + sex)
        print('birth:  ' + birth)
        print('address:' + address)
        print('idnum:  ' + idnum)
    else:
        print(info)
