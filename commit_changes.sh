#!/bin/bash

# Check git status
echo "Current git status:"
git status

# Add all changes
echo "Adding all changes..."
git add .

# Check staged changes
echo "Staged changes:"
git status

# Create commit
echo "Creating commit..."
git commit -m "Update preprocessing pipeline and environment setup

- Execute preprocessing notebook with full output and visualizations  
- Update print statements for better readability and formatting
- Add imbalanced-learn dependency to env.yaml for SMOTE functionality
- Complete preprocessing pipeline execution with all 29 cells
- Generated comprehensive output including validation results
- Ready for machine learning model development phase"

# Push to remote
echo "Pushing to remote..."
git push

echo "All changes committed and pushed successfully!" 