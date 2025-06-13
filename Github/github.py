import os
import shutil
import hashlib
import time

BASE_DIR = os.path.dirname(__file__)
OBJECTS_DIR = os.path.join(BASE_DIR, "objects")
BRANCHES_DIR = os.path.join(BASE_DIR, "branches")
COMMITS_INDEX = os.path.join(BASE_DIR, "commits_index.txt")
BRANCHES_INDEX = os.path.join(BASE_DIR, "branches_index.txt")
COMMITS_FOLDERS = os.path.join(BASE_DIR, "commits")  # New: stores full project snapshot per commit

def ensure_dirs():
    os.makedirs(OBJECTS_DIR, exist_ok=True)
    os.makedirs(BRANCHES_DIR, exist_ok=True)
    os.makedirs(COMMITS_FOLDERS, exist_ok=True)
    if not os.path.exists(COMMITS_INDEX):
        with open(COMMITS_INDEX, "w", encoding="utf-8") as f:
            f.write("")
    if not os.path.exists(BRANCHES_INDEX):
        with open(BRANCHES_INDEX, "w", encoding="utf-8") as f:
            f.write("main\n")

def store_commit(message, diff, author="AI", branch="main", extra=None, code_lines=None, code_use=None):
    ensure_dirs()
    commit_time = time.strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"author: {author}\n"
        f"branch: {branch}\n"
        f"date: {commit_time}\n"
        f"message: {message}\n"
        f"diff:\n{diff}\n"
    )
    if code_lines:
        content += f"code_lines: {code_lines}\n"
    if code_use:
        content += f"code_use: {code_use}\n"
    if extra:
        content += f"extra:\n{extra}\n"
    sha1 = hashlib.sha1(content.encode("utf-8")).hexdigest()
    obj_path = os.path.join(OBJECTS_DIR, sha1)
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(content)
    with open(COMMITS_INDEX, "a", encoding="utf-8") as idx:
        idx.write(f"{sha1},{branch}\n")
    # Save a snapshot of mini_gpt.py in commits/<commit_message>/
    safe_msg = "".join(c if c.isalnum() or c in "-_" else "_" for c in message.strip())[:32]
    commit_folder = os.path.join(COMMITS_FOLDERS, safe_msg)
    os.makedirs(commit_folder, exist_ok=True)
    files_copied = []
    # Always copy mini_gpt.py from the project root if it exists
    mini_gpt_path = os.path.join(BASE_DIR, "mini_gpt.py")
    if os.path.isfile(mini_gpt_path):
        shutil.copy2(mini_gpt_path, os.path.join(commit_folder, "mini_gpt.py"))
        files_copied.append("mini_gpt.py")
    # Also copy mini_gpt.py from C:\Users\ADMIN\Desktop\LLM if it exists and is not the same as BASE_DIR
    external_mini_gpt = r"C:\Users\ADMIN\Desktop\LLM\mini_gpt.py"
    if os.path.abspath(BASE_DIR) != os.path.abspath(r"C:\Users\ADMIN\Desktop\LLM"):
        if os.path.isfile(external_mini_gpt):
            shutil.copy2(external_mini_gpt, os.path.join(commit_folder, "mini_gpt_project_root.py"))
            files_copied.append("mini_gpt_project_root.py")
    # Remove github_sim_usage.txt if present in the commit folder
    usage_txt = os.path.join(commit_folder, "github_sim_usage.txt")
    if os.path.exists(usage_txt):
        os.remove(usage_txt)
    # Add a file with the commit sha and metadata
    with open(os.path.join(commit_folder, "commit_sha.txt"), "w", encoding="utf-8") as f:
        f.write(f"sha: {sha1}\n")
        f.write(f"branch: {branch}\n")
        f.write(f"date: {commit_time}\n")
        f.write(f"message: {message}\n")
        f.write(f"diff: {diff}\n")
        if code_lines:
            f.write(f"code_lines: {code_lines}\n")
        if code_use:
            f.write(f"code_use: {code_use}\n")
        if extra:
            f.write(f"extra: {extra}\n")
    for item in os.listdir(BASE_DIR):
        if item in ["objects", "branches", "commits", "commits_index.txt", "branches_index.txt", "GitHub", ".env", "mini_gpt.py", "github.py", "github_sim_usage.txt"]:
            continue
        s = os.path.join(BASE_DIR, item)
        d = os.path.join(commit_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
            files_copied.append(item + "/")
        elif os.path.isfile(s):
            shutil.copy2(s, d)
            files_copied.append(item)
    print(f"Commit stored: {sha1} on branch {branch}")
    print(f"mini_gpt.py snapshot saved in: {commit_folder}")
    if files_copied:
        print("Files in commit snapshot:")
        for fname in sorted(files_copied):
            print("  -", fname)
    return sha1

def list_commits(branch=None):
    ensure_dirs()
    if not os.path.exists(COMMITS_INDEX):
        print("No commits found.")
        return
    with open(COMMITS_INDEX, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            sha1, br = parts
            if branch and br != branch:
                continue
            obj_path = os.path.join(OBJECTS_DIR, sha1)
            if os.path.exists(obj_path):
                print(f"\n--- Commit {sha1} [{br}] ---")
                with open(obj_path, "r", encoding="utf-8") as obj:
                    print(obj.read())

def remove_commit(sha1):
    ensure_dirs()
    obj_path = os.path.join(OBJECTS_DIR, sha1)
    branch = None
    # Find the branch for this commit
    if os.path.exists(COMMITS_INDEX):
        with open(COMMITS_INDEX, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(sha1):
                    parts = line.strip().split(",")
                    if len(parts) > 1:
                        branch = parts[1]
                    break
    if os.path.exists(obj_path):
        os.remove(obj_path)
        print(f"Removed commit object: {sha1}")
    else:
        print(f"Commit object {sha1} not found.")
    # Update the index file
    if os.path.exists(COMMITS_INDEX):
        with open(COMMITS_INDEX, "r", encoding="utf-8") as f:
            lines = [line for line in f if not line.startswith(sha1)]
        with open(COMMITS_INDEX, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("Updated commits_index.txt.")
    # Also remove from branch folder if branch is known
    if branch:
        branch_path = os.path.join(BRANCHES_DIR, branch)
        branch_obj = os.path.join(branch_path, sha1)
        if os.path.exists(branch_obj):
            os.remove(branch_obj)
            print(f"Removed commit object: {sha1} from branch '{branch}'.")
    # Remove the commit folder snapshot (by name pattern)
    obj_path = os.path.join(OBJECTS_DIR, sha1)
    commit_folder = None
    if os.path.exists(obj_path):
        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("message:"):
                    msg = line[len("message:"):].strip()
                    safe_msg = "".join(c if c.isalnum() or c in "-_" else "_" for c in msg)[:32]
                    commit_folder = os.path.join(COMMITS_FOLDERS, safe_msg)
                    break
    if commit_folder and os.path.exists(commit_folder):
        shutil.rmtree(commit_folder)
        print(f"Removed commit folder snapshot: {commit_folder}")

def update_commit(sha1, new_message=None, new_diff=None):
    ensure_dirs()
    obj_path = os.path.join(OBJECTS_DIR, sha1)
    branch = None
    # Find the branch for this commit
    if os.path.exists(COMMITS_INDEX):
        with open(COMMITS_INDEX, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(sha1):
                    parts = line.strip().split(",")
                    if len(parts) > 1:
                        branch = parts[1]
                    break
    if not os.path.exists(obj_path):
        print(f"Commit object {sha1} not found.")
        return
    with open(obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if new_message and line.startswith("message:"):
            new_lines.append(f"message: {new_message}\n")
        elif new_diff and line.startswith("diff:"):
            new_lines.append(f"diff:\n{new_diff}\n")
        else:
            new_lines.append(line)
    with open(obj_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Commit {sha1} updated.")
    # Also update in branch folder if branch is known
    if branch:
        branch_path = os.path.join(BRANCHES_DIR, branch)
        branch_obj = os.path.join(branch_path, sha1)
        if os.path.exists(branch_obj):
            with open(branch_obj, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Commit {sha1} updated in branch '{branch}'.")
    # Update the commit folder snapshot (simulate by copying current mini_gpt.py)
    # Find the folder by commit message (from the commit object)
    obj_path = os.path.join(OBJECTS_DIR, sha1)
    commit_folder = None
    if os.path.exists(obj_path):
        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("message:"):
                    msg = line[len("message:"):].strip()
                    safe_msg = "".join(c if c.isalnum() or c in "-_" else "_" for c in msg)[:32]
                    commit_folder = os.path.join(COMMITS_FOLDERS, safe_msg)
                    break
    if commit_folder:
        if os.path.exists(commit_folder):
            shutil.rmtree(commit_folder)
        os.makedirs(commit_folder, exist_ok=True)
        mini_gpt_path = os.path.join(BASE_DIR, "mini_gpt.py")
        if os.path.isfile(mini_gpt_path):
            shutil.copy2(mini_gpt_path, os.path.join(commit_folder, "mini_gpt.py"))
        print(f"mini_gpt.py snapshot updated in: {commit_folder}")

def create_branch(branch):
    ensure_dirs()
    branch_path = os.path.join(BRANCHES_DIR, branch)
    os.makedirs(branch_path, exist_ok=True)
    # Add to branches index if not present
    with open(BRANCHES_INDEX, "r+", encoding="utf-8") as f:
        branches = f.read().splitlines()
        if branch not in branches:
            f.write(branch + "\n")
    print(f"Branch '{branch}' created.")

def push(branch="main"):
    ensure_dirs()
    branch_path = os.path.join(BRANCHES_DIR, branch)
    os.makedirs(branch_path, exist_ok=True)
    # Copy all commits for this branch to the branch folder
    with open(COMMITS_INDEX, "r", encoding="utf-8") as f:
        for line in f:
            sha1, br = line.strip().split(",")
            if br == branch:
                src = os.path.join(OBJECTS_DIR, sha1)
                dst = os.path.join(branch_path, sha1)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                # Also copy the commit folder snapshot
                commit_folder = os.path.join(COMMITS_FOLDERS, sha1)
                branch_commit_folder = os.path.join(branch_path, f"{sha1}_snapshot")
                if os.path.exists(commit_folder):
                    if os.path.exists(branch_commit_folder):
                        shutil.rmtree(branch_commit_folder)
                    shutil.copytree(commit_folder, branch_commit_folder)
    print(f"Pushed branch '{branch}' to local GitHub simulation.")

def pull(branch="main"):
    ensure_dirs()
    branch_path = os.path.join(BRANCHES_DIR, branch)
    if not os.path.exists(branch_path):
        print(f"Branch '{branch}' does not exist.")
        return
    # Copy all commits from branch folder to objects (simulate pull)
    for fname in os.listdir(branch_path):
        src = os.path.join(branch_path, fname)
        dst = os.path.join(OBJECTS_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print(f"Pulled branch '{branch}' into local objects.")

if __name__ == "__main__":
    import sys
    ensure_dirs()
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "commit":
            msg = sys.argv[2] if len(sys.argv) > 2 else "No message"
            diff = sys.argv[3] if len(sys.argv) > 3 else ""
            branch = sys.argv[4] if len(sys.argv) > 4 else "main"
            # Optionally accept code_lines and code_use as arguments 5 and 6
            code_lines = sys.argv[5] if len(sys.argv) > 5 else None
            code_use = sys.argv[6] if len(sys.argv) > 6 else None
            store_commit(msg, diff, branch=branch, code_lines=code_lines, code_use=code_use)
        elif cmd == "list":
            branch = sys.argv[2] if len(sys.argv) > 2 else None
            list_commits(branch)
        elif cmd == "remove":
            if len(sys.argv) < 3:
                print("Usage: python github.py remove <commit_sha1>")
            else:
                remove_commit(sys.argv[2])
        elif cmd == "update":
            if len(sys.argv) < 3:
                print("Usage: python github.py update <commit_sha1> [new_message] [new_diff]")
            else:
                sha1 = sys.argv[2]
                new_message = sys.argv[3] if len(sys.argv) > 3 else None
                new_diff = sys.argv[4] if len(sys.argv) > 4 else None
                update_commit(sha1, new_message, new_diff)
        elif cmd == "branch":
            if len(sys.argv) < 3:
                print("Usage: python github.py branch <branch_name>")
            else:
                create_branch(sys.argv[2])
        elif cmd == "push":
            branch = sys.argv[2] if len(sys.argv) > 2 else "main"
            push(branch)
        elif cmd == "pull":
            branch = sys.argv[2] if len(sys.argv) > 2 else "main"
            pull(branch)
        else:
            print("Unknown command.")
    else:
        print("Usage:")
        print("  python github.py commit <message> <diff> [branch]")
        print("  python github.py list [branch]")
        print("  python github.py remove <commit_sha1>")
        print("  python github.py update <commit_sha1> [new_message] [new_diff]")
        print("  python github.py branch <branch_name>")
        print("  python github.py push [branch]")
        print("  python github.py pull [branch]")