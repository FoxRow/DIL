sudo apt-get install hugin-tools enblend
pip install -r requirements.txt

wget http://pjreddie.com/media/files/yolo.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
python yolo/yad2k.py yolo.cfg yolo.weights lib/yolo.h5
