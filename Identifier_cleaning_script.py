import os

for root, dirs, files in os.walk("capstone_project"):
    for f in files:
        if "Zone.Identifier" in f:
            os.remove(os.path.join(root, f))
