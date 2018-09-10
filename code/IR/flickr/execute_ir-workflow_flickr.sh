claim=wars
name_file=/Users/perdaugmazas/Projects/iow/data/settings/dirs_$claim.txt

while IFS= read -r line
do
	printf '%s\n' "$line"
    # python retrieve-ids_flickr.py --query $line --clss $line
    python create-urls_flickr.py --clss $line
done <"$name_file"
