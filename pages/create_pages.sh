# Loop through all .ipynb files
for nb in *.ipynb; do
    # Get the base name without extension
    base_name=$(basename "$nb" .ipynb)
    
    echo "Converting $nb → $base_name.html"
    jupyter nbconvert --to html "$nb" --output "${base_name}.html"
    
    # Move the generated HTML to ./pages
    mv "${base_name}.html" ./pages/
done

echo "✅ Conversion complete! All HTML files are in ../pages/"