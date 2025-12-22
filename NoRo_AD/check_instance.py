import os, cv2, numpy as np, glob

base = "/media/vip-dell/HC/_output4/Town01/SUNNY_GLARE_DAY"  # 한 시나리오 폴더로 테스트
panos = sorted(glob.glob(os.path.join(base, "panoptic", "frame_*_panopticId.png")))[:10]

for p in panos:
    pan = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # uint16
    inst_used = (pan % 1000).astype(np.uint16)
    train = (pan // 1000).astype(np.uint16)

    # 통계
    
    u_inst = np.unique(inst_used)
    zero_ratio = np.mean(inst_used == 0)
    print(os.path.basename(p),
          f"inst_used: max={inst_used.max()} uniq={len(u_inst)} zero_ratio={zero_ratio:.3f}",
          f"trainId range: [{train.min()}, {train.max()}]")

    # 안전 체크
    assert inst_used.max() <= 999, "inst_used가 999를 초과함"
    assert set(np.unique(train)).issubset(set(range(19)) | {255}), "trainId 범위 이상"
print("✅ panoptic/instance/trainId 모두 OK")
