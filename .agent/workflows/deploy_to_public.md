---
description: Update the public repository with the latest code (clean history)
---
# Deploy to Public (Clean History)

Use this workflow to update `github.com/qrak/LLM_trader` with your latest code from `main`, while keeping the public commit history clean (single "latest" commit or squashed history).

> [!NOTE]
> Since the public repository has a squashed history, you cannot simply `git merge`. You must use this "Snapshot" method.

1.  **Preparation**: Ensure your local `main` branch is up to date and clean.
    ```bash
    git checkout main
    git pull origin main
    ```

2.  **Create Clean Snapshot**: Create a temporary orphan branch.
    ```bash
    git checkout --orphan public-release
    git add .
    git commit -m "Update: $(Get-Date -Format 'yyyy-MM-dd')"
    ```

3.  **Push to Public**: Force push to overwrite the public master branch.
    ```bash
    git push -f public public-release:master
    ```

4.  **Cleanup**: Delete the temporary branch.
    ```bash
    git checkout main
    git branch -D public-release
    ```
