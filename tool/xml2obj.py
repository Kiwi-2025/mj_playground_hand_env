import os
import xml.etree.ElementTree as ET

def mujoco_mesh_to_obj(vertex_str, vertex_normals_str, face_str, obj_filename):
    # 解析顶点数据
    vertices = [float(x) for x in vertex_str.split()]
    vertices = [vertices[i:i+3] for i in range(0, len(vertices), 3)]

    # 解析法线数据
    vertex_normals = [float(x) for x in vertex_normals_str.split()]
    vertex_normals = [vertex_normals[i:i+3] for i in range(0, len(vertex_normals), 3)]

    # 解析面数据
    faces = [int(x) for x in face_str.split()]
    faces = [faces[i:i+3] for i in range(0, len(faces), 3)]
    
    # 写入OBJ文件
    with open(obj_filename, 'w') as f:
        # 写入名称
        f.write(f"# {obj_filename}\n\n")
        
        # 写入顶点
        f.write("# List of vertices\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")

        # 写入顶点法线
        f.write("# List of vertex normals\n")
        for vn in vertex_normals:
            f.write(f"vn {vn[0]} {vn[1]} {vn[2]}\n")
        f.write("\n")

        # 写入面（OBJ索引从1开始）
        f.write("# List of faces\n")
        for face in faces:
            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        f.write("\n")

    print(f"OBJ file '{obj_filename}' created successfully.")

if __name__ == "__main__":
    # 获取当前脚本文件的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义相对路径
    meshes_dir = os.path.join(script_dir, "../xmls/meshes")
    objs_dir = os.path.join(script_dir, "../xmls/objs")

    # 确保输出目录存在
    os.makedirs(objs_dir, exist_ok=True)

    fingers = ["thumb", "index", "middle", "ring", "little"]
    parts = ["base_link", "link_1", "link_2", "link_3"]

    for finger in fingers:
        for part in parts:
            xml_file = os.path.join(meshes_dir, f"{finger}_{part}.xml")
            obj_file = os.path.join(objs_dir, f"{finger}_{part}.obj")

            # 检查 XML 文件是否存在
            if not os.path.exists(xml_file):
                print(f"File not found: {xml_file}")
                continue

            # 解析 XML 文件
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # 提取 <mesh> 元素的属性
                mesh = root.find("mesh")
                if mesh is not None:
                    vertex_str = mesh.get("vertex", "")
                    vertex_normals_str = mesh.get("normal", "")
                    face_str = mesh.get("face", "")

                    # 转换为 OBJ 文件
                    mujoco_mesh_to_obj(vertex_str, vertex_normals_str, face_str, obj_file)
                else:
                    print(f"No <mesh> element found in {xml_file}.")
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")

