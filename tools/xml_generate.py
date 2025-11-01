import os      
import numpy as np
import trimesh
from shapely.geometry import Polygon
from dataclasses import dataclass
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

current_dir = os.path.dirname(__file__)
@dataclass
class LinkParams:
    """链节参数类"""
    top_base: float
    bottom_base: float
    height: float
    prism_height: float
    hole_y: float = 0.008  
    hole_distance: float = 0.004  
    r: float = 0.01 

@dataclass
class JointParams:
    """关节参数类"""
    name: str
    pos: List[float]
    axis: List[float] = (0, 0, 1)
    stiffness: float = .1
    damping: float = 0.001
    springref: float = 0.0

@dataclass
class SiteParams:
    """节点参数类"""
    name: str
    pos: List[float]
    size: float = 0.001
    rgba: List[float] = (0, 0.9, 0, 1)

@dataclass
class FingerConfig:
    """手指整体配置类"""
    filter_parent: bool = True
    link_rgba: List[float] = (1, 1, 1, 1)
    base_pos:List[float]=(0,0,0)
    base_euler: List[float] = (0,0,0)
    link_params: List[LinkParams]=None

    base_ten_len=0.015
    # 关节参数配置
    joints: Dict[str, JointParams] = None
    sites_config : Dict[str, List[SiteParams]]=None
    finger_name:str="thumb"
    path_max = [
        ("t0", "t12"),
        ("t12", "t1_1_1"),
        ("t1_1_1", "t1_1_2"),
        ("t1_1_2", "t1_2_1"),
        ("t1_2_1", "t1_2_2"),
        ("t1_2_2", "t1_3")
    ]

    path_min = [
            ("t0", "t12"),
            ("t12", "t1_1_1"),
            ("t1_1_1", "t1_1_2"),
            ("t1_2_1", "t1_2_2")
    ]
    stl_data=None
    
    def create_solid_extruded_trapezoid(self,params) -> trimesh.Trimesh:
        """创建实体的等腰梯形体"""

        trapezoid_2d = [
            (-params.bottom_base / 2, -params.height / 2),
            (params.bottom_base / 2, -params.height / 2),
            (params.top_base / 2, params.height / 2),
            (-params.top_base / 2, params.height / 2)
        ]

        polygon = Polygon(trapezoid_2d)
        extruded_mesh = trimesh.creation.extrude_polygon(polygon, params.prism_height)
        return extruded_mesh


    def export_to_stl(mesh: trimesh.Trimesh, filename: str) -> None:
        """将 Mesh 导出为 STL 文件"""
        mesh.export(filename)
        print(f"STL 文件已保存到 {filename}")


    def generate_sites_config(self) -> Dict[str, List[SiteParams]]:
        """生成所有节点的配置

        Args:
            link_params: 链节参数列表
        """
        if self.finger_name=="thumb":
            return {
                "base_link": [
                    SiteParams("ty_1_1", [0, 0, self.link_params[0].r]),
                    SiteParams("ty_2_1", [0, 0, -self.link_params[0].r]),
                    SiteParams("tx_1_1", [0, self.link_params[0].r, 0]),
                    SiteParams("tx_2_1", [0, -self.link_params[0].r, 0]),
                    SiteParams("t12", [0, 0.01, 0], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t0", [-0.01, 0.01, 0], rgba=[0, 0, 0.9, 1])
                ],
                "link_1": [
                    SiteParams("ty_1_2", [0, 0, self.link_params[1].r]),
                    SiteParams("ty_2_2", [0, 0, -self.link_params[1].r]),
                    SiteParams("tx_1_2", [0, self.link_params[1].r, 0]),
                    SiteParams("tx_2_2", [0, -self.link_params[1].r, 0]),
                    SiteParams("t1_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_2": [
                    SiteParams("t1_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, -self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_3": [
                    SiteParams("t1_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, -self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1])
                ]
            }
        else:
            return{
                "base_link": [
                    SiteParams("t12", [0, 0, 0], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t0", [-0.01, 0, 0], rgba=[0, 0, 0.9, 1])
                ],
                "link_1": [
                    
                    SiteParams("t1_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_2": [
                    SiteParams("t1_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, -self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_3": [
                    SiteParams("t1_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, -self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1])
                ]
            }

    
    def create_solid_extruded_trapezoid(self,params) -> trimesh.Trimesh:
        """创建实体的等腰梯形体"""

        trapezoid_2d = [
            (-params.bottom_base / 2, -params.height / 2),
            (params.bottom_base / 2, -params.height / 2),
            (params.top_base / 2, params.height / 2),
            (-params.top_base / 2, params.height / 2)
        ]

        polygon = Polygon(trapezoid_2d)
        extruded_mesh = trimesh.creation.extrude_polygon(polygon, params.prism_height)
        return extruded_mesh

    def export_to_stl(mesh: trimesh.Trimesh, filename: str) -> None:
        """将 Mesh 导出为 STL 文件"""
        mesh.export(filename)
        print(f"STL 文件已保存到 {filename}")


    def generate_sites_config(self) -> Dict[str, List[SiteParams]]:
        """生成所有节点的配置

        Args:
            link_params: 链节参数列表
        """
        if self.finger_name=="thumb":
            return {
                "base_link": [
                    SiteParams("ty_1_1", [0, 0, self.link_params[0].r]),
                    SiteParams("ty_2_1", [0, 0, -self.link_params[0].r]),
                    SiteParams("tx_1_1", [0, self.link_params[0].r, 0]),
                    SiteParams("tx_2_1", [0, -self.link_params[0].r, 0]),
                    SiteParams("t12", [0, 0.01, 0], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t0", [-0.01, 0.01, 0], rgba=[0, 0, 0.9, 1])
                ],
                "link_1": [
                    SiteParams("ty_1_2", [0, 0, self.link_params[1].r]),
                    SiteParams("ty_2_2", [0, 0, -self.link_params[1].r]),
                    SiteParams("tx_1_2", [0, self.link_params[1].r, 0]),
                    SiteParams("tx_2_2", [0, -self.link_params[1].r, 0]),
                    SiteParams("t1_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_2": [
                    SiteParams("t1_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, -self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_3": [
                    SiteParams("t1_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, -self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1])
                ]
            }
        else:
            return{
                "base_link": [
                    SiteParams("t12", [0, 0, 0], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t0", [-0.01, 0, 0], rgba=[0, 0, 0.9, 1])
                ],
                "link_1": [
                    
                    SiteParams("t1_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_1", [self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_1_2", [self.link_params[1].bottom_base-self.link_params[1].hole_y, self.link_params[1].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_2": [
                    SiteParams("t1_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_1", [0.75*self.link_params[2].hole_y, self.link_params[2].hole_y, -self.link_params[2].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t1_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_2_2", [self.link_params[2].bottom_base-(0.75*self.link_params[2].hole_y), self.link_params[2].hole_y, -self.link_params[1].hole_distance], rgba=[0, 0, 0.9, 1])
                ],
                "link_3": [
                    SiteParams("t1_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1]),
                    SiteParams("t2_3", [0.75*self.link_params[3].hole_y, self.link_params[3].hole_y, -self.link_params[3].hole_distance], rgba=[0, 0, 0.9, 1])
                ]
            }

    def calculate_site_distances(self,path) -> float:
        """计算特定站点之间的总距离，使用世界坐标系

        计算路径：t12 -> t1_1_1 -> t1_1_2 -> t1_2_1 -> t1_2_2 -> t1_3

        Args:
            sites_config: 所有站点的配置信息
            link_params: 链节参数列表

        Returns:
            float: 总距离
        """
        self.sites_config=self.generate_sites_config()
        # 计算各个link的基础位置（相对于地面原点）
        link1_pos_x = self.base_ten_len  # link_1相对于base_link的x偏移
        link2_pos_x = link1_pos_x + self.link_params[1].bottom_base  # link_2相对于base_link的x偏移
        link3_pos_x = link2_pos_x + self.link_params[2].bottom_base  # link_3相对于base_link的x偏移

        # 创建站点名称到世界坐标系位置的映射
        site_positions = {}
        
        # base_link的sites
        for site in self.sites_config["base_link"]:
            site_positions[site.name] = np.array(site.pos)  # base_link的局部坐标就是世界坐标
        
        # link_1的sites
        for site in self.sites_config["link_1"]:
            local_pos = np.array(site.pos)
            world_pos = np.array([local_pos[0] + link1_pos_x, local_pos[1], local_pos[2]])
            site_positions[site.name] = world_pos
        
        # link_2的sites
        for site in self.sites_config["link_2"]:
            local_pos = np.array(site.pos)
            world_pos = np.array([local_pos[0] + link2_pos_x, local_pos[1], local_pos[2]])
            site_positions[site.name] = world_pos
        
        # link_3的sites
        for site in self.sites_config["link_3"]:
            local_pos = np.array(site.pos)
            world_pos = np.array([local_pos[0] + link3_pos_x, local_pos[1], local_pos[2]])
            site_positions[site.name] = world_pos


        # 计算总距离
        total_distance = 0
        distances = {}  # 存储每段距离的字典

        for start_site, end_site in path:
            if start_site in site_positions and end_site in site_positions:
                start_pos = site_positions[start_site]
                end_pos = site_positions[end_site]
                distance = np.linalg.norm(end_pos - start_pos)
                total_distance += distance
                distances[f"{start_site}->{end_site}"] = distance
            #else:
                #print(f"警告：找不到站点 {start_site} 或 {end_site}")

        # 打印每段距离的详细信息
        # print("\n站点间距离详情（世界坐标系）：")
        # for segment, distance in distances.items():
        #     print(f"{segment}: {distance:.4f} 米")
        # print(f"总距离: {total_distance:.4f} 米")

        return total_distance
    
    def generate_stl_files_with_centering(self,dirname) -> List[str]:
        """生成STL文件并进行中心对齐"""
        stl_filenames = []
        for i, params in enumerate(self.link_params):
            stl_filename = f"{dirname}/link_{i}.stl"
            mesh = self.create_solid_extruded_trapezoid(params)

            bottom_center = np.array([params.bottom_base / 2, params.height / 2, 0])#####################prism_height -->height
            translation_vector = bottom_center - mesh.bounds.mean(axis=0)
            mesh.apply_translation(translation_vector)

            self.export_to_stl(mesh, stl_filename)
            stl_filenames.append(stl_filename)

        return stl_filenames

    def generate_stl_data(self) -> List[str]:
        link_params=self.link_params
        #stl_filenames = []
        link_vertex=[]
        link_normal=[]
        link_face=[]
        for i, params in enumerate(link_params):
            mesh = self.create_solid_extruded_trapezoid(params)

            bottom_center = np.array([params.bottom_base / 2, params.height / 2, 0])#####################prism_height -->height
            translation_vector = bottom_center - mesh.bounds.mean(axis=0)
            mesh.apply_translation(translation_vector)
            mesh.process()  # 确保法线被计算
            link_vertex.append(mesh.vertices.flatten())
            link_face.append(mesh.faces.flatten())
            link_normal.append(mesh.vertex_normals.flatten())

        #print(link_vertex[1])
        self.stl_data=(link_vertex,link_normal,link_face)
        
    def marker_generate(self, nx, ny, dx, dy) -> list:
        cx = 0
        cy = 0
        size = np.array([0.0003, 0.0003, 0.001])
        x_start = cx - (nx - 1) * dx / 2
        y_start = cy - (ny - 1) * dy / 2
        x_end = cx + (nx - 1) * dx / 2
        y_end = cy + (ny - 1) * dy / 2

        x_coords = np.linspace(x_start, x_end, nx)
        y_coords = np.linspace(y_start, y_end, ny)
        points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(nx, ny, 2)

        tip_attribute = {
            "friction": "0.7 0.05 0.0002",
            "solref": "0.02 1.5",
            "solimp": "0.9 0.99 0.001",
            "condim": "3"
        }

        marker_elements = []
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                x, y = points[i][j]
                body_elem = ET.Element(
                    "body",
                    name=f"{self.finger_name}_markerB_{i}_{j}",
                    pos=f"{x} {y} -0.002"
                )
                # 生成盒状弹簧振子geom元素
                # geom_elem = ET.Element(
                #     "geom",
                #     name=f"{self.finger_name}_markerG_{i}_{j}",
                #     type="box",
                #     conaffinity="1",
                #     contype="1",
                #     rgba=".9 0 0 .9",
                #     size=f"{size[0]} {size[1]} {size[2]}",
                #     mass=".0001",
                #     friction=tip_attribute["friction"],
                #     solref=tip_attribute["solref"],
                #     solimp=tip_attribute["solimp"],
                #     condim=tip_attribute["condim"]
                # )

                # 生成球状弹簧振子geom元素
                geom_elem = ET.Element(
                    "geom",
                    name=f"{self.finger_name}_markerG_{i}_{j}",
                    type="sphere",
                    conaffinity="1",
                    contype="1",
                    rgba=".9 0 0 .9",
                    size=f"{size[0]}",
                    mass=".0001",
                    friction=tip_attribute["friction"],
                    solref=tip_attribute["solref"],
                    solimp=tip_attribute["solimp"],
                    condim=tip_attribute["condim"]
                )
                body_elem.append(geom_elem)
                marker_elements.append(body_elem)
        return marker_elements
    
    def generate_sites_xml(self, link_name) -> list:
        sites = self.sites_config.get(link_name, [])
        site_elements = []
        for site in sites:
            elem = ET.Element(
                "site",
                name=f"{self.finger_name}_{site.name}",
                pos=f"{site.pos[0]} {site.pos[1]} {site.pos[2]}",
                size=str(site.size),
                rgba=f"{site.rgba[0]} {site.rgba[1]} {site.rgba[2]} {site.rgba[3]}"
            )
            site_elements.append(elem)
        return site_elements

    def generate_finger_collision_box_xml(self, link_num) -> list:
        rgba = "0 0 1 0"
        rgba_visible = "1 1 1 .3"
        d = 0.001
        attribute = {"density": "200", "friction": "0.7 0.05 0.0002", "solref": "0.02 1.5", "solimp": "0.9 0.99 0.001"}
        sphere_radius = self.link_params[link_num].prism_height * 0.5
        sphere_pos_x = self.link_params[link_num].bottom_base
        sphere_pos_y = self.link_params[link_num].height * 0.6
        cylinder_pos = 0.02
        if link_num < 2:
            rad = 0.785
        else:
            rad = 0.925

        geoms = []
        if link_num < 3:
            # Center box
            geoms.append(ET.Element(
                "geom",
                type="box",
                pos=f"{self.link_params[link_num].bottom_base/2} {self.link_params[link_num].height/2} 0",
                size=f"{self.link_params[link_num].top_base/2} {self.link_params[link_num].height/2} {self.link_params[link_num].prism_height/2}",
                name=f"{self.finger_name}_link_{link_num}_box_center",
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
                rgba=rgba,
            ))
            # Slope 1
            geoms.append(ET.Element(
                "geom",
                type="box",
                pos=f"{self.link_params[link_num].bottom_base*0.75+self.link_params[link_num].top_base*0.25-d*np.sin(rad)-d*np.cos(rad)} {self.link_params[link_num].height/2-d*np.cos(rad)+d*np.sin(rad)} 0",
                size=f"{(self.link_params[link_num].bottom_base-self.link_params[link_num].top_base)*0.5*0.5/np.cos(rad)-d*np.tan(np.pi*0.5-rad)} {d} {self.link_params[link_num].prism_height/2}",
                euler=f"0 0 {-rad}",
                name=f"{self.finger_name}_link_{link_num}_box_slope1",
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
                rgba=rgba,
            ))
            # Slope 2
            geoms.append(ET.Element(
                "geom",
                type="box",
                pos=f"{self.link_params[link_num].bottom_base*0.25-self.link_params[link_num].top_base*0.25+d*np.sin(rad)+d*np.cos(rad)} {self.link_params[link_num].height/2+d*np.cos(rad)-d*np.sin(rad)} 0",
                size=f"{(self.link_params[link_num].bottom_base-self.link_params[link_num].top_base)*0.5*0.5/np.cos(rad)-d*np.tan(np.pi*0.5-rad)} {d} {self.link_params[link_num].prism_height/2}",
                euler=f"0 0 {rad}",
                name=f"{self.finger_name}_link_{link_num}_box_slope2",
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
                rgba=rgba,
            ))
            # Bottom box
            geoms.append(ET.Element(
                "geom",
                type="box",
                pos=f"{self.link_params[link_num].bottom_base*0.5} {d} 0",
                size=f"{self.link_params[link_num].bottom_base*0.5-2*d} {d} {self.link_params[link_num].prism_height/2}",
                euler="0 0 0",
                name=f"{self.finger_name}_link_{link_num}_box_bottom",
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
                rgba=rgba,
            ))
        else:
            # Center box
            geoms.append(ET.Element(
                "geom",
                type="box",
                pos=f"{self.link_params[link_num].bottom_base/2} {self.link_params[link_num].height/2} 0",
                size=f"{self.link_params[link_num].top_base/2} {self.link_params[link_num].height/2} {self.link_params[link_num].prism_height/2}",
                name=f"{self.finger_name}_link_{link_num}_box_center",
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
                rgba=rgba,
            ))
            # Slope 2
            geoms.append(ET.Element(
                "geom",
                type="box",
                pos=f"{self.link_params[link_num].bottom_base*0.25-self.link_params[link_num].top_base*0.25+d*np.sin(rad)+d*np.cos(rad)} {self.link_params[link_num].height/2+d*np.cos(rad)-d*np.sin(rad)} 0",
                size=f"{(self.link_params[link_num].bottom_base-self.link_params[link_num].top_base)*0.5*0.5/np.cos(rad)-d*np.tan(np.pi*0.5-rad)} {d} {self.link_params[link_num].prism_height/2}",
                euler=f"0 0 {rad}",
                name=f"{self.finger_name}_link_{link_num}_box_slope2",
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
                rgba=rgba,
            ))
            # Sphere
            geoms.append(ET.Element(
                "geom",
                type="sphere",
                pos=f"{sphere_pos_x} {sphere_pos_y} 0",
                size=f"{sphere_radius}",
                rgba=rgba_visible,
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
            ))
            # Cylinder
            geoms.append(ET.Element(
                "geom",
                type="cylinder",
                euler="0 1.57 0",
                pos=f"{cylinder_pos} {sphere_pos_y} 0",
                size=f"{sphere_radius} {sphere_pos_x-cylinder_pos}",
                rgba=rgba_visible,
                density=attribute["density"], friction=attribute["friction"], 
                solref=attribute["solref"], solimp=attribute["solimp"],
            ))
        return geoms

    def generate_finger_xml(self):
        ten_len_max=self.calculate_site_distances(self.path_max)
        ten_len_min=self.calculate_site_distances(self.path_min)
        self.generate_stl_data()
        
        # Generate XML structure
        # Generate meshes XML
        xml_meshes = []
        xml_meshes.append(ET.Element(
            "mesh",
            name=f"{self.finger_name}_base_link",
            vertex=" ".join(map(str, self.stl_data[0][0])),
            normal=" ".join(map(str, self.stl_data[1][0])),
            face=" ".join(map(str, self.stl_data[2][0]))
        ))
        xml_meshes.append(ET.Element(
            "mesh",
            name=f"{self.finger_name}_link_1",
            vertex=" ".join(map(str, self.stl_data[0][1])),
            normal=" ".join(map(str, self.stl_data[1][1])),
            face=" ".join(map(str, self.stl_data[2][1]))
        ))
        xml_meshes.append(ET.Element(
            "mesh",
            name=f"{self.finger_name}_link_2",
            vertex=" ".join(map(str, self.stl_data[0][2])),
            normal=" ".join(map(str, self.stl_data[1][2])),
            face=" ".join(map(str, self.stl_data[2][2]))
        ))
        xml_meshes.append(ET.Element(
            "mesh",
            name=f"{self.finger_name}_link_3",
            vertex=" ".join(map(str, self.stl_data[0][3])),
            normal=" ".join(map(str, self.stl_data[1][3])),
            face=" ".join(map(str, self.stl_data[2][3]))
        ))

        if self.finger_name == "thumb":
            xml_body = ET.Element("body",name=f"{self.finger_name}_base_link",pos=" ".join(map(str, self.base_pos)),euler=" ".join(map(str, self.base_euler)))
            ET.SubElement(xml_body, "geom", conaffinity="0", contype="0", type="sphere", size=".002")
            # TODO：Add sites xml
            sites_xml = self.generate_sites_xml("base_link")
            for site_elem in sites_xml:
                xml_body.append(site_elem)

            # link_1
            link_1_body = ET.Element(
            "body",
            name=f"{self.finger_name}_link_1",
            pos=f"{self.base_ten_len} 0 0"
            )
            ET.SubElement(link_1_body, "joint",
            name=f"{self.finger_name}_{self.joints['universal_1'].name}",
            pos=" ".join(map(str, self.joints['universal_1'].pos)),
            axis="0 0 1",
            damping=str(self.joints['universal_1'].damping)
            )
            ET.SubElement(link_1_body, "joint",
            name=f"{self.finger_name}_{self.joints['universal_2'].name}",
            pos=" ".join(map(str, self.joints['universal_2'].pos)),
            axis="0 1 0",
            damping=str(self.joints['universal_2'].damping)
            )
            
            # TODO:Add collision box
            collision_box_xml = self.generate_finger_collision_box_xml(link_num=1)
            for geom in collision_box_xml:
                link_1_body.append(geom)

            ET.SubElement(link_1_body, "geom", conaffinity="0", contype="0", type="mesh", rgba=" ".join(map(str, self.link_rgba)), mesh=f"{self.finger_name}_link_1")
            
            # TODO:Add sites
            sites_xml = self.generate_sites_xml("link_1")
            for site_elem in sites_xml:
                link_1_body.append(site_elem)

            # link_2
            link_2_body = ET.Element("body",name=f"{self.finger_name}_link_2",pos=f"{self.link_params[1].bottom_base} 0 0")
            ET.SubElement(  link_2_body, "joint",
                            name=f"{self.finger_name}_{self.joints['joint_1'].name}",
                            pos=" ".join(map(str, self.joints['joint_1'].pos)),
                            axis=" ".join(map(str, self.joints['joint_1'].axis)),
                            stiffness=str(self.joints['joint_1'].stiffness),
                            springref=str(self.joints['joint_1'].springref),
                            damping=str(self.joints['joint_1'].damping),
                            range="0 3.14",
            )

            collision_box_xml = self.generate_finger_collision_box_xml(link_num=2)
            for geom in collision_box_xml:
                link_2_body.append(geom)

            ET.SubElement(link_2_body, "geom", conaffinity="0", contype="0", type="mesh", rgba=" ".join(map(str, self.link_rgba)), mesh=f"{self.finger_name}_link_2")
            
            sites_xml = self.generate_sites_xml("link_2")
            for site_elem in sites_xml:
                link_2_body.append(site_elem)

            # link_3
            link_3_body = ET.Element("body",name=f"{self.finger_name}_link_3",pos=f"{self.link_params[2].bottom_base} 0 0")
            ET.SubElement(link_3_body, "joint",
            name=f"{self.finger_name}_{self.joints['joint_2'].name}",
            pos=" ".join(map(str, self.joints['joint_2'].pos)),
            axis=" ".join(map(str, self.joints['joint_2'].axis)),
            stiffness=str(self.joints['joint_2'].stiffness),
            springref=str(self.joints['joint_2'].springref),
            damping=str(self.joints['joint_2'].damping),
            range="0 3.14",
            )

            collision_box_xml = self.generate_finger_collision_box_xml(link_num=3)
            for geom in collision_box_xml:
                link_3_body.append(geom)

            ET.SubElement(link_3_body, "geom", conaffinity="0", contype="0", type="mesh", rgba=" ".join(map(str, self.link_rgba)), mesh=f"{self.finger_name}_link_3")

            sites_xml = self.generate_sites_xml("link_3")
            for site_elem in sites_xml:
                link_3_body.append(site_elem)

            # fingertip
            fingertip_body = ET.Element("body",name=f"{self.finger_name}_fingertip",pos=f"0.02 {self.link_params[3].height} 0",euler="1.57 0 0")
            ET.SubElement(fingertip_body, "geom", conaffinity="0", contype="0", type="sphere", pos="0 0 0", size=".001", rgba=".9 0 0 .1")
            
            markers_xml = self.marker_generate(9, 9, 0.001, 0.001)
            for marker in markers_xml:
                fingertip_body.append(marker)

            # Nest bodies
            link_3_body.append(fingertip_body)
            link_2_body.append(link_3_body)
            link_1_body.append(link_2_body)
            xml_body.append(link_1_body)
            xml_body = [xml_body]

            # 生成 tendon 节点
            xml_tendons = []
            # x1 tendon
            tendon_x1 = ET.Element("spatial", name=f"{self.finger_name}_tendon_x1", width=".0005", rgba=".9 .0 .0 1")
            tendon_x1.append(ET.Element("site", site=f"{self.finger_name}_tx_1_1"))
            tendon_x1.append(ET.Element("site", site=f"{self.finger_name}_tx_1_2"))
            xml_tendons.append(tendon_x1)
            # x2 tendon
            tendon_x2 = ET.Element("spatial", name=f"{self.finger_name}_tendon_x2", width=".0005", rgba=".9 .0 .0 1")
            tendon_x2.append(ET.Element("site", site=f"{self.finger_name}_tx_2_2"))
            tendon_x2.append(ET.Element("site", site=f"{self.finger_name}_tx_2_1"))
            xml_tendons.append(tendon_x2)
            # y1 tendon
            tendon_y1 = ET.Element("spatial", name=f"{self.finger_name}_tendon_y1", width=".0005", rgba="0 .9 .0 1")
            tendon_y1.append(ET.Element("site", site=f"{self.finger_name}_ty_1_1"))
            tendon_y1.append(ET.Element("site", site=f"{self.finger_name}_ty_1_2"))
            xml_tendons.append(tendon_y1)
            # y2 tendon
            tendon_y2 = ET.Element("spatial", name=f"{self.finger_name}_tendon_y2", width=".0005", rgba="0 .9 .0 1")
            tendon_y2.append(ET.Element("site", site=f"{self.finger_name}_ty_2_2"))
            tendon_y2.append(ET.Element("site", site=f"{self.finger_name}_ty_2_1"))
            xml_tendons.append(tendon_y2)
            # main tendon
            tendon_main = ET.Element("spatial", name=f"{self.finger_name}_tendon", width=".0005", rgba="0 0 .9 1", limited="true", range=f"{ten_len_min} {ten_len_max}", solreflimit=".001 1", solimplimit="0.99 0.999 0.00001 0.1 2")
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t0"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t12"))
            tendon_main.append(ET.Element("pulley", divisor="2"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t12"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t1_1_1"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t1_1_2"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t1_2_1"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t1_2_2"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t1_3"))
            tendon_main.append(ET.Element("pulley", divisor="2"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t12"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t2_1_1"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t2_1_2"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t2_2_1"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t2_2_2"))
            tendon_main.append(ET.Element("site", site=f"{self.finger_name}_t2_3"))
            xml_tendons.append(tendon_main)

            # 生成 actuator 节点
            xml_actuators = []
            xml_actuators.append(ET.Element("motor", name=f"{self.finger_name}_tendon", tendon=f"{self.finger_name}_tendon", ctrllimited="true", ctrlrange="-20 0"))
            xml_actuators.append(ET.Element("position", name=f"{self.finger_name}_tendon_x1", tendon=f"{self.finger_name}_tendon_x1", forcerange="-50 0", forcelimited="true", ctrllimited="true", ctrlrange=f" 0 {2*self.base_ten_len}", kp="10000", dampratio="1.5"))
            xml_actuators.append(ET.Element("position", name=f"{self.finger_name}_tendon_x2", tendon=f"{self.finger_name}_tendon_x2", forcerange="-50 0", forcelimited="true", ctrllimited="true", ctrlrange=f" 0 {2*self.base_ten_len}", kp="10000", dampratio="1.5"))
            xml_actuators.append(ET.Element("position", name=f"{self.finger_name}_tendon_y1", tendon=f"{self.finger_name}_tendon_y1", forcerange="-50 0", forcelimited="true", ctrllimited="true", ctrlrange=f" 0 {2*self.base_ten_len}", kp="10000", dampratio="1.5"))
            xml_actuators.append(ET.Element("position", name=f"{self.finger_name}_tendon_y2", tendon=f"{self.finger_name}_tendon_y2", forcerange="-50 0", forcelimited="true", ctrllimited="true", ctrlrange=f" 0 {2*self.base_ten_len}", kp="10000", dampratio="1.5"))

        else:
            xml_swing_joint, xml_swing_actuator = None, None
            if self.finger_name == "index":
                xml_swing_joint = ET.Element("joint", name=f"{self.finger_name}_swing", axis="0 -1 0", range="0 0.5", damping="0.2", )
                xml_swing_actuator = ET.Element("position", name=f"{self.finger_name}_swing", joint=f"{self.finger_name}_swing", ctrllimited="true", ctrlrange="0 0.5", kp="1000", dampratio="2")
            elif self.finger_name == "ring":
                xml_swing_joint = ET.Element("joint", name=f"{self.finger_name}_swing", axis="0 1 0", range="0 0.5", damping="0.2", )
                xml_swing_actuator = ET.Element("position", name=f"{self.finger_name}_swing", joint=f"{self.finger_name}_swing", ctrllimited="true", ctrlrange="0 0.5", kp="1000", dampratio="2")
            elif self.finger_name == "little":
                xml_swing_joint = ET.Element("joint", name=f"{self.finger_name}_swing", axis="0 1 0", range="0 0.75", damping="0.2", )
                xml_swing_actuator = ET.Element("position", name=f"{self.finger_name}_swing", joint=f"{self.finger_name}_swing", ctrllimited="true", ctrlrange="0 0.75", kp="1000", dampratio="2")

            ## Create body XML, body xml 是嵌套的
            xml_body = ET.Element("body", name=f"{self.finger_name}_base_link", pos=" ".join(map(str, self.base_pos)), euler=" ".join(map(str, self.base_euler)))
            if xml_swing_joint is not None:
                xml_body.append(xml_swing_joint)
                ET.SubElement(xml_body, "geom", conaffinity="0", contype="0", type="sphere", size=".002")

            for site_elem in self.generate_sites_xml("base_link"):
                xml_body.append(site_elem)

            # Create link 1 XML
            link_1_body = ET.Element("body", name=f"{self.finger_name}_link_1", pos=f"{self.base_ten_len} 0 0")
            ET.SubElement(link_1_body, "joint", 
                name=f"{self.finger_name}_{self.joints['joint_0'].name}", 
                pos=" ".join(map(str, self.joints['joint_0'].pos)), 
                axis=" ".join(map(str, self.joints['joint_0'].axis)), 
                stiffness=str(self.joints['joint_0'].stiffness), 
                springref=str(self.joints['joint_0'].springref), 
                damping=str(self.joints['joint_0'].damping), 
                range="0 3.14", 
            )

            for geom in self.generate_finger_collision_box_xml(link_num=1):
                link_1_body.append(geom)

            ET.SubElement(link_1_body, "geom", conaffinity="0", contype="0", type="mesh", rgba=" ".join(map(str, self.link_rgba)), mesh=f"{self.finger_name}_link_1")
            for site_elem in self.generate_sites_xml("link_1"):
                link_1_body.append(site_elem)

            # Create link 2 XML
            link_2_body = ET.Element("body", name=f"{self.finger_name}_link_2", pos=f"{self.link_params[1].bottom_base} 0 0")
            ET.SubElement(link_2_body, "joint", 
                name=f"{self.finger_name}_{self.joints['joint_1'].name}", 
                pos=" ".join(map(str, self.joints['joint_1'].pos)), 
                axis=" ".join(map(str, self.joints['joint_1'].axis)), 
                stiffness=str(self.joints['joint_1'].stiffness), 
                springref=str(self.joints['joint_1'].springref), 
                damping=str(self.joints['joint_1'].damping), 
                range="0 3.14", 
            )
            for geom in self.generate_finger_collision_box_xml(link_num=2):
                link_2_body.append(geom)

            ET.SubElement(link_2_body, "geom", conaffinity="0", contype="0", type="mesh", rgba=" ".join(map(str, self.link_rgba)), mesh=f"{self.finger_name}_link_2")

            for site_elem in self.generate_sites_xml("link_2"):
                link_2_body.append(site_elem)

            # Create link 3 XML
            link_3_body = ET.Element("body", name=f"{self.finger_name}_link_3", pos=f"{self.link_params[2].bottom_base} 0 0")
            ET.SubElement(
                link_3_body,
                "joint",
                name=f"{self.finger_name}_{self.joints['joint_2'].name}",
                pos=" ".join(map(str, self.joints['joint_2'].pos)),
                axis=" ".join(map(str, self.joints['joint_2'].axis)),
                stiffness=str(self.joints['joint_2'].stiffness),
                springref=str(self.joints['joint_2'].springref),
                damping=str(self.joints['joint_2'].damping),
                range="0 1.3"
            )

            for geom in self.generate_finger_collision_box_xml(link_num=3):
                link_3_body.append(geom)

            ET.SubElement(link_3_body, "geom", conaffinity="0", contype="0", type="mesh", rgba=" ".join(map(str, self.link_rgba)), mesh=f"{self.finger_name}_link_3")

            for site in self.generate_sites_xml("link_3"):
                link_3_body.append(site)

            # Create fingertip XML
            fingertip_body = ET.Element("body", name=f"{self.finger_name}_fingertip", pos=f"0.02 {self.link_params[3].height} 0", euler="1.57 0 0")
            ET.SubElement(fingertip_body, "geom", conaffinity="0", contype="0", type="sphere", pos="0 0 0", size=".001", rgba=".9 0 0 .1")
            for marker in self.marker_generate(9, 9, 0.001, 0.001):
                fingertip_body.append(marker)

            link_3_body.append(fingertip_body)
            link_2_body.append(link_3_body)
            link_1_body.append(link_2_body)
            xml_body.append(link_1_body)
            xml_body = [xml_body]

            # Create tendon XML, tendon xml 只有一个
            xml_tendons = []
            tendon = ET.Element("spatial", name=f"{self.finger_name}_tendon", width=".0005", rgba="0 0 .9 1", limited="true", range=f"{ten_len_min} {ten_len_max}", solreflimit=".001 1", solimplimit="0.99 0.999 0.00001 0.1 2")
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t0"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t12"))
            tendon.append(ET.Element("pulley", divisor="2"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t12"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t1_1_1"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t1_1_2"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t1_2_1"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t1_2_2"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t1_3"))
            tendon.append(ET.Element("pulley", divisor="2"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t12"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t2_1_1"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t2_1_2"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t2_2_1"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t2_2_2"))
            tendon.append(ET.Element("site", site=f"{self.finger_name}_t2_3"))

            xml_tendons.append(tendon)

            # Create actuator XML， 有多个
            xml_actuators = [
                # ET.Element("position", name=f"{self.finger_name}_tendon", tendon=f"{self.finger_name}_tendon", forcerange="-20 0", forcelimited="true", ctrllimited="true", ctrlrange=f"{ten_len_min*0.3} {ten_len_max}", kp="1000", dampratio="1.5"),
                ET.Element("motor", name=f"{self.finger_name}_tendon", tendon=f"{self.finger_name}_tendon", ctrllimited="true", ctrlrange="-20 0"),
            ]
            if xml_swing_actuator is not None:
                xml_actuators.append(xml_swing_actuator)

        return xml_meshes,xml_body,xml_tendons,xml_actuators

def create_mujoco_xml(hand: List[FingerConfig], obj_name, file_name, output_file=False) -> None:
    """
    创建 MuJoCo XML 文件的函数。该函数按照 MuJoCo XML 文件的结构，依次添加各个元素，生成完整的手部模型描述。
    以下是添加 XML 元素的顺序和每个部分的作用：

    1. **根元素 `<mujoco>`**：
       - 创建根元素 `<mujoco>`，并设置模型名称为 "hand"。
       - 这是 XML 文件的顶层节点，所有其他元素都嵌套在其中。

    2. **全局设置 `<compiler>` 和 `<option>`**：
       - 添加 `<compiler>` 元素，用于设置编译选项，例如角度单位（`angle="radian"`）。
       - 添加 `<option>` 元素，用于设置仿真选项，例如积分器类型（`integrator="implicitfast"`）。

    3. **尺寸设置 `<size>`**：
       - 添加 `<size>` 元素，用于设置仿真模型的最大关节数（`njmax`）和最大接触数（`nconmax`）。

    4. **默认设置 `<default>`**：
       - 添加 `<default>` 元素，用于定义几何体（`geom`）的默认属性，例如弹性和阻尼参数（`solref` 和 `solimp`）。

    5. **仿真选项 `<option>`**：
       - 添加额外的 `<option>` 子元素，例如禁用父过滤器（`filterparent="disable"`）。

    6. **接触设置 `<contact>`**：
       - 添加 `<contact>` 元素，用于定义接触相关的全局设置。

    7. **世界几何体和手部模型 `<worldbody>`**：
       - 创建 `<worldbody>` 元素，作为所有物体的容器。
       - 在 `<worldbody>` 中添加手掌（`palm`）和其他手指的几何体和关节。

    8. **手指模型 `<body>`**：
       - 遍历 `hand` 列表，为每个手指生成对应的 `<body>` 元素。
       - 每个手指的 `<body>` 包括关节（`joint`）、几何体（`geom`）和站点（`site`）。

    9. **绳索定义 `<tendon>`**：
       - 添加 `<tendon>` 元素，用于定义手指的绳索路径。
       - 每个绳索由多个 `<site>` 元素和 `pulley` 元素组成。

    10. **控制器定义 `<actuator>`**：
        - 添加 `<actuator>` 元素，用于定义控制器。
        - 每个控制器与关节或绳索关联，并设置控制范围（`ctrlrange`）和增益（`kp`）。

    11. **约束定义 `<equality>`**：
        - 添加 `<equality>` 元素，用于定义关节之间的耦合关系。

    12. **文件输出**：
        - 如果 `output_file=True`，将生成的 XML 写入文件。
        - 文件路径为当前目录下的 `{obj_name}.xml`。
    """

    # 创建根元素
    mujoco = ET.Element("mujoco", model="hand")

    # 添加简单孤立子元素
    ET.SubElement(mujoco, "compiler", angle="radian")
    ET.SubElement(mujoco, "option", integrator="implicitfast")
    ET.SubElement(mujoco, "size", njmax="5000", nconmax="10000")

    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "geom", solref="0.001 1", solimp="0.99 0.999 0.0001")
    
    option = ET.SubElement(mujoco, "option")
    ET.SubElement(option, "flag", filterparent ="disable")

    contact = ET.SubElement(mujoco, "contact")
    excludes = [
        {"body1": "palm", "body2": "thumb_link_1"},
        {"body1": "palm", "body2": "index_link_1"},
        {"body1": "palm", "body2": "middle_link_1"},
        {"body1": "palm", "body2": "ring_link_1"},
        {"body1": "palm", "body2": "little_link_1"},
    ]
    for exclude in excludes:
        ET.SubElement(contact, "exclude", body1=exclude["body1"], body2=exclude["body2"])
    
    # 添加 asset 子节点
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(mujoco, "option", gravity="0 0 -9.81")

    for finger in hand:
        xml_meshes, _, _, _ = finger.generate_finger_xml()
        for mesh in xml_meshes:
            asset.append(mesh)
    ET.SubElement(asset,"texture",type="2d",name="groundplane",builtin="checker",mark="edge",rgb1="1 1 1",rgb2="0 0 0",markrgb="0 0 0",width="300",height="300")
    ET.SubElement(asset,"material",name="groundplane",texture="groundplane",texuniform="true",texrepeat="5 5",reflectance="0")

    # 添加 worldbody 节点
    worldbody = ET.SubElement(mujoco, "worldbody")
    palm = ET.SubElement(worldbody, "body", name="palm", pos="0 0 0.15", euler="3.14 3.14 0")
    joints = [
        {"name": "palm_slide_x", "type": "slide", "axis": "1 0 0", "range": "-.5 .5"},
        {"name": "palm_slide_y", "type": "slide", "axis": "0 1 0", "range": "-.5 .5"},
        {"name": "palm_slide_z", "type": "slide", "axis": "0 0 1", "range": "-.2 .12"},
        {"name": "palm_rotate_x", "type": "hinge", "axis": "1 0 0", "range": "-1.57 1.57"},
        {"name": "palm_rotate_y", "type": "hinge", "axis": "0 1 0", "range": "-1.57 1.57"},
        {"name": "palm_rotate_z", "type": "hinge", "axis": "0 0 1", "range": "-3.14 3.14"},
    ]
    for joint in joints:
        ET.SubElement(palm,"joint",name=joint["name"],type=joint["type"],axis=joint["axis"],range=joint["range"],limited="true",solreflimit=".001 1",solimplimit="0.97 0.99 0.00001 0.1 2",damping="0.2")
    ET.SubElement(palm, "geom", type="box", size="0.045 0.045 0.002", rgba="1 1 1 1", density="200")
    ET.SubElement(palm,"geom",type="box",size="0.0375 0.045 0.01",pos="-0.0075 0 0.012",rgba="1 1 1 1",density="200")

    # 添加手指的 body 节点
    for finger in hand:
        _, xml_bodies, _, _ = finger.generate_finger_xml()
        for xml_body in xml_bodies:
            palm.append(xml_body)

    # 添加 tendon 节点
    tendon = ET.SubElement(mujoco, "tendon")
    for finger in hand:
        _, _, xml_tendons, _ = finger.generate_finger_xml()
        for xml_tendon in xml_tendons:
            tendon.append(xml_tendon)

    # 添加 actuator 节点
    actuator = ET.SubElement(mujoco, "actuator")
    for finger in hand:
        _, _, _, xml_actuators = finger.generate_finger_xml()
        for xml_actuator in xml_actuators:
            actuator.append(xml_actuator)
    
    actuators = [
        {"name": "palm_slide_x", "joint": "palm_slide_x", "ctrlrange": "-.1 .1"},
        {"name": "palm_slide_y", "joint": "palm_slide_y", "ctrlrange": "-.1 .1"},
        {"name": "palm_slide_z", "joint": "palm_slide_z", "ctrlrange": "-.2 .1"},
        {"name": "palm_rotate_x", "joint": "palm_rotate_x", "ctrlrange": "-.2 .2"},
        {"name": "palm_rotate_y", "joint": "palm_rotate_y", "ctrlrange": "-.2 .2"},
        {"name": "palm_rotate_z", "joint": "palm_rotate_z", "ctrlrange": "-3.14 3.14"},
    ]
    for act in actuators:
        ET.SubElement(actuator,"position",name=act["name"],joint=act["joint"],ctrllimited="true",ctrlrange=act["ctrlrange"],kp="1000",dampratio="1.5",)

    # 添加 equality 节点
    # equality = ET.SubElement(mujoco, "equality")
    # ET.SubElement(equality,"joint", name="ring_little", joint1="ring_swing", joint2="little_swing", polycoef="0 0.67 0 0 0")

    # 格式化 XML
    xml_str = ET.tostring(mujoco, encoding="unicode")
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

    # 写入文件
    if output_file:
        current_dir = os.path.dirname(__file__)
        output_path = os.path.join(current_dir, file_name)
        # output_path = os.path.join(current_dir, "hand.xml")
        with open(output_path, "w") as f:
            f.write(xml_str)
        print(f"Mujoco XML 文件已保存到 {output_path}")

    return xml_str

def calculate_bottom_base(top_base: float, height: float, is_first_two: bool) -> float:
        """
        根据不同的链节计算 bottom_base 的值

        Args:
            top_base: 顶部宽度
            height: 高度
            is_first_two: 是否是前两个链节

        Returns:
            float: 计算得到的底部宽度
        """
        if is_first_two:
            # 前两个链节：bottom_base = top_base + 2*height
            return top_base + 2 * height
        else:
            # 后两个链节：bottom_base = top_base + (3*height/2)
            return top_base + (3 * height / 2)

def finger_para_generate(finger_name,link_top_base,link_height,link_prism_height,joint_damping,joint_stiffness,base_pos,base_euler):
    """
    生成手指参数
    """
    if finger_name=="thumb":
        joint_params = {
                "universal_1": JointParams(
                    name="universal_1",
                    pos=[0, 0, 0],
                    damping=0.1
                ),
                "universal_2": JointParams(
                    name="universal_2",
                    pos=[0, 0, 0],
                    damping=0.1
                ),
                "joint_1": JointParams(
                    name="joint_1",
                    pos=[0, 0, 0],
                    stiffness=joint_stiffness[0],
                    damping=joint_damping
                ),
                "joint_2": JointParams(
                    name="joint_2",
                    pos=[0, 0, 0],
                    stiffness=joint_stiffness[1],
                    damping=joint_damping
                )
            }
    else:
        joint_params = {
                "joint_0":JointParams(
                    name="joint_0",
                    pos=[0, 0, 0],
                    stiffness=joint_stiffness[0],
                    damping=joint_damping
                ),
                "joint_1": JointParams(
                    name="joint_1",
                    pos=[0, 0, 0],
                    stiffness=joint_stiffness[1],
                    damping=joint_damping
                ),
                "joint_2": JointParams(
                    name="joint_2",
                    pos=[0, 0, 0],
                    stiffness=joint_stiffness[2],
                    damping=joint_damping
                )
            }
    link_params = [
        # base_link
        LinkParams(
            top_base=link_top_base[0],
            height=link_height,
            bottom_base=calculate_bottom_base(link_top_base[0], link_height, True),  # 自动计算
            prism_height=link_prism_height,
            r=0.01
        ),
        # link_1
        LinkParams(
            top_base=link_top_base[1],
            height=link_height,
            bottom_base=calculate_bottom_base(link_top_base[1], link_height, True),  # 自动计算
            prism_height=link_prism_height,

            r=0.01
        ),
        # link_2
        LinkParams(
            top_base=link_top_base[2],
            height=link_height,
            bottom_base=calculate_bottom_base(link_top_base[2], link_height, False),  # 自动计算
            prism_height=link_prism_height,

        ),
        # link_3
        LinkParams(
            top_base=link_top_base[3],
            height=link_height,
            bottom_base=calculate_bottom_base(link_top_base[3], link_height, False),  # 自动计算
            prism_height=link_prism_height,

        )
    ]

    finger= FingerConfig(
        link_params=link_params,
        joints=joint_params,
        finger_name=finger_name,
        base_pos=base_pos,
        base_euler=base_euler,
    )

    return finger

def hand_para_generate(thumb_params,index_params,middle_params,ring_params,little_params,file_name, output_file=False):
    
    thumb=finger_para_generate(**thumb_params)
    index=finger_para_generate(**index_params)
    middle=finger_para_generate(**middle_params)
    ring=finger_para_generate(**ring_params)
    little=finger_para_generate(**little_params)
    
    hand = [thumb,index,middle,ring,little]

    obj_name="cube"
    xml_str = create_mujoco_xml(hand,obj_name=obj_name,file_name=file_name,output_file=output_file)
    return xml_str

def params_to_dict(params):
    thumb_params = {
        "finger_name": "thumb",
        "link_top_base": [0.02, params[0,0], params[0,1], params[0,2]],
        "link_height": params[0,3],
        "link_prism_height": params[0,4],
        "joint_damping": 0.2,
        "joint_stiffness": params[0,5]*params[0,6:8],
        "base_pos": params[0,9:12],
        "base_euler": (params[0,12], params[0,13], params[0,14])
    }
    index_params = {
        "finger_name": "index",
        "link_top_base": [0.02, params[1,0], params[1,1], params[1,2]],
        "link_height": params[1,3],
        "link_prism_height": params[1,4],
        "joint_damping": 0.2,
        "joint_stiffness": params[1,5]*params[1,6:9],
        "base_pos": params[1,9:12],
        "base_euler": (params[1,12], params[1,13], params[1,14])
    }
    middle_params = {
        "finger_name": "middle",
        "link_top_base": [0.02, params[2,0], params[2,1], params[2,2]],
        "link_height": params[2,3],
        "link_prism_height": params[2,4],
        "joint_damping": 0.2,
        "joint_stiffness": params[2,5]*params[2,6:9],
        "base_pos": params[2,9:12],
        "base_euler": (params[2,12], params[2,13], params[2,14])
    }
    ring_params = {
        "finger_name": "ring",
        "link_top_base": [0.02, params[3,0], params[3,1], params[3,2]],
        "link_height": params[3,3],
        "link_prism_height": params[3,4],
        "joint_damping": 0.2,
        "joint_stiffness": params[3,5]*params[3,6:9],
        "base_pos": params[3,9:12],
        "base_euler": (params[3,12], params[3,13], params[3,14])
    }
    little_params = {
        "finger_name": "little",
        "link_top_base": [0.02, params[4,0], params[4,1], params[4,2]],
        "link_height": params[4,3],
        "link_prism_height": params[4,4],
        "joint_damping": 0.2,
        "joint_stiffness": params[4,5]*params[4,6:9],
        "base_pos": params[4,9:12],
        "base_euler": (params[4,12], params[4,13], params[4,14])
    }
    return thumb_params, index_params, middle_params, ring_params, little_params

def main():

    
    # thumb=finger_para_generate(finger_name="thumb",link_top_base=[0.02,0.02,0.01,0.01],link_height=0.01,link_prism_height=0.015,joint_damping=0.2,joint_stiffness=[0.1,0.1],base_pos=[-.035,-.02,-.005],base_euler=(-0.785,0,-1.57))
    # index=finger_para_generate(finger_name="index",link_top_base=[0.02,0.02,0.01,0.01],link_height=0.01,link_prism_height=0.015,joint_damping=0.2,joint_stiffness=[0.1,0.1,0.1],base_pos=[0.015,-0.03,.005],base_euler=(1.57,0,0))
    # middle=finger_para_generate(finger_name="middle",link_top_base=[0.02,0.02,0.01,0.01],link_height=0.01,link_prism_height=0.015,joint_damping=0.2,joint_stiffness=[0.1,0.1,0.1],base_pos=[0.02,-0.01,.005],base_euler=(1.57,0,0))
    # ring=finger_para_generate(finger_name="ring",link_top_base=[0.02,0.02,0.01,0.01],link_height=0.01,link_prism_height=0.015,joint_damping=0.2,joint_stiffness=[0.1,0.1,0.1],base_pos=[0.015,0.01,.005],base_euler=(1.57,0,0))
    # little=finger_para_generate(finger_name="little",link_top_base=[0.02,0.02,0.01,0.01],link_height=0.01,link_prism_height=0.015,joint_damping=0.2,joint_stiffness=[0.1,0.1,0.1],base_pos=[0.005,0.03,.005],base_euler=(1.57,0,0))
    

    # hand = [thumb,index,middle,ring,little]
    params=np.array([
        [0.02,0.01,0.01,  0.01,  0.015,  0.1,1,1,0,  -.035, -.02, -.005,  -0.785, 0, -1.57],
        [0.02,0.01,0.01,  0.01,  0.015,  0.1,1,1,1,  0.015, -0.03, .005,  1.57, 0, 0],
        [0.02,0.01,0.01,  0.01,  0.015,  0.1,1,1,1,  0.02, -0.01, .005,   1.57, 0, 0],
        [0.02,0.01,0.01,  0.01,  0.015,  0.1,1,1,1,  0.015, 0.01, .005,   1.57, 0, 0],
        [0.02,0.01,0.01,  0.01,  0.015,  0.1,1,1,1,  0.005, 0.03, .005,   1.57, 0, 0]
    ])
    
    #ten_len_xy = thumb.base_ten_len
    thumb,index,middle,ring,little=params_to_dict(params)
    file_name = "sphere_marker_hand.xml"
    xml_str = hand_para_generate(thumb,index,middle,ring,little,file_name=file_name,output_file=True)

if __name__ == "__main__":
    main()










    