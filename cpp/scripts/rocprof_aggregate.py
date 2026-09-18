import re
import sys

path = sys.argv[1]
tot = {}
cnt = {}
for line in open(path, encoding="utf-8", errors="replace"):
    if not line.startswith("dispatch"):
        continue
    m = re.search(r"time\((\d+),(\d+),(\d+),(\d+)\)", line)
    n = re.search(r'kernel-name\("([^"]+)"\)', line)
    if not m or not n:
        continue
    d = int(m.group(3)) - int(m.group(2))
    k = n.group(1)[:44]
    tot[k] = tot.get(k, 0) + d
    cnt[k] = cnt.get(k, 0) + 1
g = sum(tot.values())
nd = sum(cnt.values())
print(f"dispatches={nd} kernel_time={g/1e9:.2f}s mean_kernel={g/nd/1e3:.1f}us")
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:6]:
    print(f"{v/1e6:8.0f}ms {cnt[k]:6d}x {v/cnt[k]/1e3:6.1f}us/each  {k}")
