name_source=$1
scale=$2
target=$3
PATH_CORPORA=/Users/perdaugmazas/Projects/iow/data/DP/vision/$name_source/selected/$target/
# TODO: - move raw images to selected and apply 
# 		- Fix output path

source activate opencv
for entry in $PATH_CORPORA/*/; do
	path="$entry"
	name=$(basename "$path")
	echo "$name"
	python create-descs_lookup.py --lookup "$name" --source $name_source -n 250 --scale $scale
done
source deactivate