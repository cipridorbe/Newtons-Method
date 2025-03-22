echo "Deleting build/"
mv build/_deps .
rm -rf build/
mv _deps/ build/ 
echo "Creating build/" 
cmake -S . -B build/
