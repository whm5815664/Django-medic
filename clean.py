from pathlib import Path

p = Path("requirements.txt")
lines = p.read_text(encoding="utf-8").splitlines()

cleaned = []
for line in lines:
    s = line.strip()
    if " @ file:///" in s or " @ file://" in s:
        continue
    cleaned.append(line)

p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
print("已删除所有带 file:// 链接的整行")