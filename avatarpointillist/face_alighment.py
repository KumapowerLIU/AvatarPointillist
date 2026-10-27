# import cv2
# import numpy as np
# import mediapipe as mp

# # 使用 Mediapipe 提取人脸关键点
# def get_face_landmarks(image):
#     mp_face_mesh = mp.solutions.face_mesh
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.multi_face_landmarks:
#             return None
#         landmarks = results.multi_face_landmarks[0].landmark
#         h, w = image.shape[:2]
#         # 选择一些关键点：左眼右眼鼻尖嘴角（仿 affine 变换需要3~5个）
#         indices = [33, 263, 1, 61, 291]  # 左眼右眼鼻尖左嘴角右嘴角
#         points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices], dtype=np.float32)
#         return points

# # 对齐函数
# def align_face(src_img, dst_img):
#     src_pts = get_face_landmarks(src_img)
#     dst_pts = get_face_landmarks(dst_img)
#     if src_pts is None or dst_pts is None:
#         print("❌ 人脸检测失败")
#         return None
#     # 计算仿射变换矩阵
#     matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
#     aligned = cv2.warpAffine(src_img, matrix, (dst_img.shape[1], dst_img.shape[0]))
#     return aligned

# # 加载图像
# src_img = cv2.imread("/nas5/liuhongyu/AR_Results/ners_test_data/inference_imgs/musk/00000_00.png")  # 源图
# dst_img = cv2.imread("/nas5/liuhongyu/AR_Results/ners_test_data/inference_imgs/076_FREE/00097_08.png")  # 目标图

# # 对齐
# aligned_img = align_face(src_img, dst_img)

# # 显示结果
# if aligned_img is not None:
#     # cv2.imshow("Aligned Face", aligned_img)
#     # cv2.imshow("Target", dst_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     cv2.imwrite("aligned_face.png", aligned_img)
# else:
#     print("对齐失败")


import os
base_dir = '/nas5/liuhongyu/AR_Results/ners_test_data/motion_seqs'
save_dir = '/nas5/liuhongyu/AR_Results/baseline/portrait4d/out_results'
input_ref_dir = "/nas5/liuhongyu/AR_Results/ners_test_data/inference_imgs/"
all_ids = os.listdir(input_ref_dir)
ar_out_base = '/nas5/liuhongyu/AR_Results/ners_test_data/ar_out/ar_out'
all_ids.sort()
all_ids = ['093_EMO-2', '148_EMO-1', '106_EMO-4', '097_EMO-3']
# all_ids_allready = ['416_FREE', '425_EXP-9', '434_EXP-8', '453_EXP-5', '467_EXP-4', '510_EXP-3', '516_EMO-4', '522_EXP-2', '538_EMO-1']
for ids in all_ids:
    # if ids in all_ids_allready:
    #     continue
    input_dir_base = os.path.join(input_ref_dir, ids )  
    image_name = [k for k in os.listdir(input_dir_base) if k.endswith('png')] [0]
    id_real = ids.split('_')[0]
    ar_input = os.path.join(ar_out_base, id_real)  
    # motion_seq = "/nas5/liuhongyu/AR_Results/ners_test_data/motion_seqs/076_FREE/" 
    motion_seq = os.path.join(base_dir, '076_FREE')
    cmd = f'CUDA_VISIBLE_DEVICES=1 ./scripts_gstrans/inference.sh  configs/arlam-20k-hongyu-use-ar-binding-arfeature-largeoffset-test.yaml /nas5/liuhongyu/AR_Results/ners_test_data/ours_weights/arlam-20k-hongyu-use-ar-binding-arfeature-largeoffset-newanimation-finetune/006500/  {input_dir_base}/  {motion_seq}/  {image_name} {ar_input}'
    # cmd = f'CUDA_VISIBLE_DEVICES=2 ./scripts/inference.sh  configs/lam-20k.yaml  /nas5/liuhongyu/AR_Results/ners_test_data/weights/lam-20k_step-25k/  {input_dir_base}/  {motion_seq}/'
 
    print(cmd)
    os.system(cmd)