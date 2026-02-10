---
description: Workflow for implementing a GitHub issue on a feature branch with PR
---

## Issue Branch Workflow

### 1. Ensure main is up to date
// turbo
```bash
git checkout main && git pull origin main
```

### 2. Create a feature branch
Name the branch using the convention: `fix/<issue>-<short-name>` for bugs, `feat/<issue>-<short-name>` for enhancements.
```bash
git checkout -b <branch-name>
```

### 3. Implement the changes
- Follow the acceptance criteria in the GitHub issue
- Add comments explaining non-obvious logic
- Keep changes scoped to the issue

### 4. Stage and commit
Use conventional commit messages referencing the issue number.
// turbo
```bash
git add -A && git commit -m "<type>: <description> (closes #<issue>)"
```
Types: `fix` for bugs, `feat` for enhancements, `refactor` for restructuring, `docs` for documentation.

### 5. Push the branch
// turbo
```bash
git push -u origin <branch-name>
```

### 6. Create a Pull Request
// turbo
```bash
gh pr create --title "<type>: <description> (#<issue>)" --body "Closes #<issue>" --base main
```

### 7. After merge, clean up
// turbo
```bash
git checkout main && git pull origin main && git branch -d <branch-name>
```
