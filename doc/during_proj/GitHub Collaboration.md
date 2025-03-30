# GitHub Collaboration

## Branch Organization

整个项目应该有多个分支。首先 `main`是作为公开的发布分支，每次合并其他的分支来进行更新。`dev`是工作分支，可以进行 overnight 的提交。我们每次工作都需要创建一个自己本地的分支，例如 `feature/feature_name`。其中 `feature`是一个类别或命名空间，类似目录结构，用于更好的分支管理。

## Local Development process

完整的工作流程如下：

1. **从 `dev` 创建个人分支**：
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **在个人分支上进行开发**。

3. **完成后，将个人分支合并回 `dev`**：
   
   完成开发并准备合并回 `dev` 分支的步骤如下：
   
   1. **确保个人分支是最新的** [1]：
      
      ```bash
      git checkout feature/your-feature-name
      git pull origin dev
      ```
      
   2. **解决任何可能的冲突** [2]，然后提交解决方案：
      
      ```bash
      git add .
      git commit -m "Resolve merge conflicts"
      ```
      
   3. **推送个人分支到远程仓库**：
      ```bash
      git push origin feature/your-feature-name
      ```
   
   4. **创建 Pull Request (PR)**：
      
      - 在 GitHub 上，创建一个 PR，从 `feature/your-feature-name` 合并到 `dev`。
      
   5. **经过审查后合并 PR**：
      - 在 PR 被批准后，将其合并到 `dev` 分支。
   
   6. **删除个人分支**：在 PR 之后就可以删除个人分支了，因为 PR 是基于远程分支的，一直存储在远程仓库中，直到被合并或者被关闭。
      
      ```bash
      git checkout dev
      git branch -d feature/your-feature-name
      git push origin --delete feature/your-feature-name
      ```
      
      因为是可以 push overnight，本地的个人分支不需要立即删除。这是一个可选步骤。
   
   这样可以确保你的工作顺利合并到 `dev` 分支中。

这样可以保持团队协作的有序性和代码的稳定性。

## Further Information

[1] 情景再现：
		在工作之前，`dev`上有 `a.txt`, `c.txt`。当你在本地修改了 `c.txt`，而别人添加了 `b.txt` 并推送到 `dev`，如果你在分支 `local_feat` 上执行 `git pull origin dev`，会发生以下情况：

1. **获取并合并更改**：
   - Git 会从远程 `dev` 分支获取最新的更改，包括新添加的 `b.txt`。

2. **合并操作**：
   - **自动合并**：如果你的修改不与 `dev` 的更改冲突（例如，`c.txt` 的修改和 `b.txt` 的添加无关），Git 会自动合并这些更改。
   - **文件更新**：`b.txt` 会被添加到你的本地分支，而你对 `c.txt` 的修改会被保留。

3. **没有冲突**：
   - 由于 `b.txt` 是新文件，且你的修改在 `c.txt`，通常不会有冲突。

4. **完成合并**：
   - 如果一切顺利，Git 会完成合并，你的分支 `local_feat` 将包含最新的 `a.txt`、`b.txt` 和修改过的 `c.txt`。

这样，你的本地分支就会同步到 `dev` 的最新状态，同时保留你的更改。

[2] 这种冲突只有在 pull 下来的版本和本地版本都修改了同一个文件的时候产生。需要手动修改然后再重新 `add`