# Instructions to Push Code to GitHub

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name it "unified-dta-project" (or any name you prefer)
   - Make sure to leave "Initialize this repository with a README" unchecked
   - Click "Create repository"

2. Copy the repository URL (it should look like `https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git`)

3. Run these commands in the terminal (replace YOUR_USERNAME and YOUR_REPO_NAME with your actual values):

```bash
cd unified_dta_project/unified_dta_project
git remote remove origin  # Remove any existing origin
git remote add origin https://github.com/iAMv1/unified-dta-project
git branch -M main
git push -u origin main
```

If you get an authentication error, you may need to:

1. Create a Personal Access Token on GitHub:
   - Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
   - Click "Generate new token (classic)"
   - Give it a name and select the "repo" scope
   - Click "Generate token"
   - Copy the token (you won't see it again)

2. Use the token to authenticate:
   ```bash
   git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

Or you can clone the repository using GitHub CLI or Desktop and then copy the files over.