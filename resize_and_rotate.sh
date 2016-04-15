inpdir=$1
outdir=$2
mkdir -p $outdir
for file in $inpdir*.jpg; do
    convert "$file" -resize 10% -rotate 90 "$outdir$(basename ${file%.jpg})".jpg
done
