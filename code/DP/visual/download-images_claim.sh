claim=$2
name_file=/Users/perdaugmazas/Projects/iow/data/settings/dirs_$claim.txt
name_source=$1

while IFS= read -r line
do
	printf '%s\n' "$line"
    python download-images.py --source $name_source --clss $line
done <"$name_file"