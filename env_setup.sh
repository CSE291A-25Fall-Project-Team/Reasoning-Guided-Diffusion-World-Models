cd packages
git clone -b robocasa https://github.com/ARISE-Initiative/robomimic && pip install -e robomimic 
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite 
git clone https://github.com/robocasa/robocasa && pip install -e robocasa

# python robocasa/robocasa/scripts/download_kitchen_assets.py && \
rm -rf robocasa/robocasa/models/assets
mkdir -p robocasa/robocasa/models/assets && unzip asset.zip -d robocasa/robocasa/models/assets
python robocasa/robocasa/scripts/setup_macros.py

cd ..
pip install -r requirements.txt