#!/bin/bash
target="combined-deployment.yaml"

files=(
    "postgres-pvc"
    "postgres-deployment"
    "postgres-service"
    "predictor-deployment"
)

first_file=${files[0]}
last_file=${files[-1]}
echo "Adding files:"
for file in ${files[@]}; do
    filename="$file.yaml"
    echo "    $filename"
    if [[ "$file" == "$first_file" ]]; then
        cat "$filename" > "$target"
    else
        cat "$filename" >> "$target"
    fi
    if [[ "$file" != "$last_file" ]]; then
        echo >> "$target"
        echo "---" >> "$target"
    fi
done

echo "Created $target"
