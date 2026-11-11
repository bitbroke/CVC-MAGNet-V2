import subprocess
from pathlib import Path

def main():
    root = Path("C:/Ankshit/CVC-MAGNet-V2")
    res = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    print(f"Total untracked / modified items: {len(lines)}")
    
    too_large = []
    normal_files = []
    for l in lines:
        rel = l[3:].strip('"')
        p = root / rel
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > 50.0:  # GitHub blocks > 100MB and warns > 50MB
                too_large.append((rel, size_mb))
            else:
                normal_files.append((rel, size_mb))
        elif p.is_dir():
            # Check directory size
            dir_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
            if dir_size > 50.0:
                too_large.append((rel + " (dir)", dir_size))
            else:
                normal_files.append((rel + " (dir)", dir_size))

    print(f"\nFiles/Dirs > 50 MB ({len(too_large)} items):")
    for r, sz in too_large:
        print(f"  {r}: {sz:.1f} MB")
        
    print(f"\nCandidate normal files to commit ({len(normal_files)} items):")
    for r, sz in normal_files[:20]:
        print(f"  {r}: {sz:.2f} MB")
    if len(normal_files) > 20:
        print(f"  ... and {len(normal_files) - 20} more.")

if __name__ == "__main__":
    main()
