# Cloning FindMe Repository (1st step)

Because this is a **private GitHub repository**, you **cannot use your GitHub password** to clone it.  
Instead, you must use a **Personal Access Token (PAT)**.

Follow the steps below carefully.

---

## Step 1: Copy the Repository HTTPS URL

https://github.com/HameedJHD/FindMe.git

---

## Step 2: Create a Personal Access Token (PAT)

1. Go to **GitHub → Settings**
2. Click **Developer settings**
3. Click **Personal access tokens**
4. Select **Tokens (classic)**
5. Click **Generate new token (classic)**
6. Configure the token:
   - **Note:** `any name you chose`
   - **Expiration:** any time you want
   - **Scopes:**  `repo` (required)

7. Click **Generate token**
8. **Copy the token immediately** (you will not be able to see it again)

---

## Step 3: Clone the Repository

Open a terminal (Command Prompt, PowerShell, or Terminal):

```bash
cd path/to/your/folder
git clone https://github.com/HameedJHD/FindMe.git

"You'll be prompt for username and password"
"enter your username when prompt"
"Paste the Token when prompt for password"

