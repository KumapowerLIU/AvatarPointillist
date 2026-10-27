import sys
from collections import defaultdict, deque


def parse_and_separate_obj(file_path):
    """
    读取OBJ文件，分离非面数据和面数据。
    面数据将保留其原始的 v/vt/vn 字符串格式。
    """
    non_face_lines = []
    # faces 将存储为 [['v1/vt1', 'v2/vt2', 'v3/vt3'], ...]
    faces = []

    with open(file_path, 'r') as f:
        for line in f:
            # 我们只处理以 'f ' 开头的行，注意空格以避免匹配 'ffo' 等
            if line.startswith('f '):
                # parts 将是 ['f', 'v1/vt1/vn1', 'v2/vt2/vn2', 'v3/vt3/vn3']
                parts = line.strip().split()
                # 我们只处理三角形，取前三个顶点定义
                if len(parts) >= 4:
                    faces.append(parts[1:4])
                else:
                    print(f"Warning: Non-triangular face found, skipping: {line.strip()}")
            else:
                # 保留所有其他行 (v, vt, vn, o, g, usemtl, mtllib, etc.)
                non_face_lines.append(line)

    return non_face_lines, faces


def build_edge_to_face_map(faces):
    """
    根据面列表构建边到面索引的映射。
    关键：从 'v/vt/vn' 字符串中提取顶点索引 'v' 来定义边。
    """
    edge_map = defaultdict(list)
    for i, face_parts in enumerate(faces):
        try:
            # 从 'v/vt/vn' 中提取顶点索引
            v_indices = [int(p.split('/')[0]) for p in face_parts]

            # 创建规范化的边元组
            edges = [
                tuple(sorted((v_indices[0], v_indices[1]))),
                tuple(sorted((v_indices[1], v_indices[2]))),
                tuple(sorted((v_indices[2], v_indices[0]))),
            ]

            for edge in edges:
                edge_map[edge].append(i)
        except (ValueError, IndexError) as e:
            print(
                f"Warning: Could not parse vertex indices from face {i}, skipping. Error: {e}. Face parts: {face_parts}")

    return edge_map


def build_adjacency_graph(edge_map):
    """根据边到面的映射构建面邻接图。"""
    neighbors = defaultdict(set)
    for edge, face_indices in edge_map.items():
        # 如果一条边被两个面共享，则这两个面相邻
        if len(face_indices) == 2:
            f1, f2 = face_indices[0], face_indices[1]
            neighbors[f1].add(f2)
            neighbors[f2].add(f1)
    return neighbors


def bfs_face_sort(faces, neighbors):
    """
    使用广度优先搜索（BFS）对所有面进行排序。
    能处理有多个不相连部分的网格。
    """
    if not faces:
        return []

    num_faces = len(faces)
    visited = set()
    sorted_faces_indices = []

    # 循环确保所有不相连的组件都被访问到
    for start_node in range(num_faces):
        if start_node not in visited:
            queue = deque([start_node])
            visited.add(start_node)

            while queue:
                current_face_idx = queue.popleft()
                sorted_faces_indices.append(current_face_idx)

                # 对邻居排序以获得确定性的、可复现的结果
                for neighbor_idx in sorted(list(neighbors.get(current_face_idx, []))):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        queue.append(neighbor_idx)

    # 根据排序后的索引，重新排列原始的面数据
    # 这里的 faces[i] 仍然是 ['v1/vt1', 'v2/vt2', 'v3/vt3'] 的形式
    final_sorted_faces = [faces[i] for i in sorted_faces_indices]
    return final_sorted_faces


def write_sorted_obj(output_path, non_face_lines, sorted_faces):
    """将非面数据和排序后的面数据写入新的OBJ文件。"""
    with open(output_path, 'w') as f:
        # 1. 首先写入所有非面数据（v, vt, vn, g, o, ...）
        for line in non_face_lines:
            f.write(line)

        f.write("\n# --- Sorted faces below ---\n")

        # 2. 然后写入排序后的面数据
        # face 是一个列表，如 ['652/767', '654/768', '655/769']
        # ' '.join(face) 会将其变回 "652/767 654/768 655/769"
        for face in sorted_faces:
            f.write(f"f {' '.join(face)}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python face_sorter.py input.obj output_sorted.obj")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading and parsing '{input_path}'...")
    non_face_lines, faces = parse_and_separate_obj(input_path)

    if not faces:
        print("No faces found in the OBJ file. Copying file as is.")
        with open(input_path, 'r') as src, open(output_path, 'w') as dst:
            dst.write(src.read())
        return

    print(f"Found {len(faces)} faces. Building adjacency graph...")
    edge_map = build_edge_to_face_map(faces)
    neighbors = build_adjacency_graph(edge_map)

    print("Sorting faces using BFS...")
    sorted_faces = bfs_face_sort(faces, neighbors)

    print(f"Writing sorted OBJ to '{output_path}'...")
    write_sorted_obj(output_path, non_face_lines, sorted_faces)

    print(f"Successfully sorted {len(sorted_faces)} faces and saved to '{output_path}'.")


if __name__ == "__main__":
    main()