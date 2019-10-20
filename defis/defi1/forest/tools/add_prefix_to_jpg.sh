for name in *.jpeg
do
    newname="$(echo "$1$name")"
    mv "$name" "$newname"
done

for name in *.jpg
do
    newname="$(echo "$1$name")"
    mv "$name" "$newname"
done

for name in *.png
do
    newname="$(echo "$1$name")"
    mv "$name" "$newname"
done