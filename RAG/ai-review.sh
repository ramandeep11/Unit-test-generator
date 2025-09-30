#!/bin/bash

# Function to get AI feedback on code changes
get_ai_feedback() {
    local diff_content="$1"
    local file_name="$2"
    
    # Ensure diff content is non-empty before calling Ollama
    if [ -z "$diff_content" ]; then
        echo "No diff content for $file. Skipping AI feedback."
        return
    fi

    # Use Ollama to generate feedback
    feedback=$(ollama run llama3.1 "Review the following code changes for the file '$file_name' and provide feedback and relevant suggestions:

$diff_content

Please provide your feedback in a concise manner, focusing on:
1. Potential bugs or issues
2. Code style and best practices
3. Performance improvements
4. Security concerns (if any)
5. Any other relevant observations

Feedback:" 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "Error while getting AI feedback for $file_name. Please check the Ollama setup."
        return
    fi

    echo -e "\nAI Feedback for $file_name:\n$feedback\n"
}

# Get unstaged changes
unstaged_files=$(git diff --name-only)

# Get staged changes
staged_files=$(git diff --cached --name-only)

# Combine and remove duplicates
all_changed_files=$(echo -e "$unstaged_files\n$staged_files" | sort -u)

# Check if there are any changed files
if [ -z "$all_changed_files" ]; then
    echo "No changed files detected."
    exit 0
fi

# Process each changed file
for file in $all_changed_files; do
    echo "Processing file: $file"
    
    # Get the combined diff (staged and unstaged changes)
    diff_content=$(git diff HEAD -- "$file")

    # If there's no diff content, skip to the next file
    if [ -z "$diff_content" ]; then
        echo "No changes detected for $file. Skipping."
        continue
    fi
    
    # Get AI feedback
    get_ai_feedback "$diff_content" "$file"
done

echo "AI code review complete."