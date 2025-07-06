import re

# adjust these paths to where your files actually live
INPUT = r"C:\Users\17ayu\OneDrive\Desktop\ExploratoryDataAnalysis-EDA-UsingStreamlit\requirements_raw.txt"
OUTPUT = r"C:\Users\17ayu\OneDrive\Desktop\ExploratoryDataAnalysis-EDA-UsingStreamlit\requirements.txt"

# any wheel basename → real PyPI name adjustments
manual = {
    "scikit_learn":     "scikit-learn",
    "python_dateutil":  "python-dateutil",
    "rpds_py":          "rpds-py",
    "sklearn_compat":   "sklearn-compat",
    "MarkupSafe":       "MarkupSafe",  # usually the same
}

reqs = []
with open(INPUT, "r", encoding="utf-8") as f:
    lines = [l.rstrip() for l in f]

i = 0
while i < len(lines):
    line = lines[i].strip()
    # detect wheel filename
    if line.endswith(".whl"):
        wheel = line  # e.g. "altair-5.5.0-py3-none-any.whl"
        # sanity check next two lines exist
        if i + 2 < len(lines):
            hash_line = lines[i + 2].strip()
            # should be exactly 64 hex chars
            if re.fullmatch(r"[0-9a-f]{64}", hash_line):
                # extract name & version from wheel filename
                name, ver, *_ = wheel.split("-", 2)
                pkg = manual.get(name, name)
                reqs.append(f"{pkg}=={ver} --hash=sha256:{hash_line}")
            else:
                print(f"⚠️ bad hash at line {i+2}: {hash_line}")
        i += 4  # skip wheel, SHA256 header, hash, CertUtil
    else:
        i += 1

# write out
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(reqs))

print(f"✅ Wrote {len(reqs)} entries to {OUTPUT}")
