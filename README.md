# 第二代身份证信息识别
可识别身份证上所有信息：姓名，性别，民族，出生日期，住址，身份证号码。提供Docker镜像部署方式
# 依赖：
> 本项目在Ubuntu 18.04 基于tesseract 4.0 ，OpenCV3.4; 使用Python3.6进行开发<br>
> apt依赖安装：<br>
>`sudo apt install python3 python3-pip tesseract-ocr tesseract-ocr-chi-sim tzdata libsm6 libxext6 python3-tk -y` <br><br>
> Python依赖安装：<br>
>`sudo pip3 install -r idcardocr/requirements.txt`<br><br>
> tessdata配置：<br>
> `sudo cp tessdata/* /usr/share/tesseract-ocr/tessdata`<br>
# 使用方法：
> 识别本地图片<br>
> ```bash
> python3 test.py