# Clone Github Repo to Local

### 解决方案：

#### **方法 1：使用 HTTPS + 个人访问令牌（推荐）**

GitHub 现在要求使用 **个人访问令牌（Personal Access Token, PAT）** 来代替密码。

**步骤：**

1. 进入 GitHub [Token 生成页面](https://github.com/settings/tokens)。

2. 点击 **"Generate new token"**（如果有 Fine-grained tokens 选项，选它）。

3. 选择 **repo（完整访问仓库）** 权限，并 **保存生成的 Token**（只能看到一次）。

4. 运行以下命令：

   ```bash
   git clone https://<YOUR_GITHUB_USERNAME>:<YOUR_PERSONAL_ACCESS_TOKEN>@github.com/OwenCalstroy/Bi-layer-Structure.git
   ```

   ⚠️ 注意

   ：

   - `<YOUR_GITHUB_USERNAME>` 替换为你的 GitHub 用户名。
   - `<YOUR_PERSONAL_ACCESS_TOKEN>` 替换为你的访问令牌（不要包含 `<>`）。
   - 访问令牌不能包含特殊字符（如 `@`），否则需要 URL 编码。

#### **方法 2：使用 SSH Key 认证**

如果你不想使用 Token，每次都输入密码，可以改用 SSH。

**步骤：**

1. 先检查本地是否已有 SSH key：

   ```bash
   ls ~/.ssh/id_rsa.pub
   ```

   - 如果文件存在，跳过第 2 步。
   - 如果不存在，执行下一步。

2. 生成 SSH key：

   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```

   （一路回车，默认生成 `~/.ssh/id_rsa` 和 `~/.ssh/id_rsa.pub`）。

3. 添加 SSH key 到 GitHub：

   ```bash
   cat ~/.ssh/id_rsa.pub
   ```

   复制输出内容，并在 [GitHub SSH Keys](https://github.com/settings/keys) 页面添加。

4. 测试 SSH 连接：

   ```bash
   ssh -T git@github.com
   ```

   如果成功，会显示：

   ```
   Hi <YOUR_GITHUB_USERNAME>! You've successfully authenticated...
   ```

5. 现在可以使用 SSH 方式克隆：

   ```bash
   git clone git@github.com:OwenCalstroy/Bi-layer-Structure.git
   ```

------

#### **总结**

- **推荐 HTTPS + 个人访问令牌（简单，适合临时访问）**
- **SSH 适合长期使用（安全，免输入密码）**
- **不要直接用 `git clone github.com/...`，要加 `https://` 或 `git@`**

试试看，看看哪种方法适合你！🚀