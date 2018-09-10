claim=wars
name_file=/Users/perdaugmazas/Projects/iow/data/settings/dirs_$claim.txt

while IFS= read -r line
do
	printf '%s\n' "$line"
    python retrieve-album-ids_fb.py --query $line
    python extract-album-data_fb.py --query $line
    python retrieve-image-ids_fb.py --query $line  
    python extract-image-data_fb.py --query $line
    python create-urls_fb.py --query $line
done <"$name_file"