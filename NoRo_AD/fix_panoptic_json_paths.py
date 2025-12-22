# fix_panoptic_json_paths.py
import os, json, re, shutil, sys

def main(json_path, panoptic_root):
    with open(json_path, "r") as f:
        js = json.load(f)

    def candidates(fn: str):
        m = re.search(r"^(Town\d+)/(.*?)/(?:panoptic/)?frame_(\d+)_.*\.(png|jpg)$", fn, re.IGNORECASE)
        if not m:
            return [fn]
        town, scen, idx = m.group(1), m.group(2), m.group(3)
        return [
            f"{town}/{scen}/panoptic/frame_{idx}_panopticId.png",
            f"{town}/{scen}/panoptic/frame_{idx}_panopticId.jpg",
            f"{town}/{scen}/frame_{idx}_panopticId.png",
            f"{town}/{scen}/frame_{idx}_panopticId.jpg",
        ]

    changed, missing = 0, 0
    for ann in js.get("annotations", []):
        old = ann.get("file_name", "")
        # 공백/대소문자 normalize
        old_norm = old.strip()
        cands = candidates(old_norm)

        chosen = None
        for rel in cands:
            if os.path.exists(os.path.join(panoptic_root, rel)):
                chosen = rel
                break
        if chosen is None:
            # png로 표준화만
            chosen = cands[0]
            if not os.path.exists(os.path.join(panoptic_root, chosen)):
                missing += 1

        if chosen != old:
            ann["file_name"] = chosen
            changed += 1

    # 백업 & 저장
    backup = json_path + ".bak"
    shutil.copy2(json_path, backup)
    with open(json_path, "w") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)

    print(f"[DONE] patched: {changed}, still-missing: {missing}")
    if missing > 0:
        print("[HINT] missing>0이면 panoptic_root 경로나 폴더 구조(Town/Scenario/panoptic) 재확인 필요.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python fix_panoptic_json_paths.py <panoptic_json> <panoptic_root>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
