python retrieve-album-ids_fb.py --query $1
python extract-album-data_fb.py --query $1
python retrieve-image-ids_fb.py --query $1 
python extract-image-data_fb.py --query $1
python create-urls_fb.py --query $1