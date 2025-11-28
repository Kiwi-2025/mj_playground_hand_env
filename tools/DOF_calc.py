# FILE: DOF_calc.py
import mujoco
import os

def calculate_dof(xml_str):
    """
    Load a MuJoCo XML model and calculate its Degrees of Freedom (DOF).
    
    Args:
        xml_path (str): Path to the MuJoCo XML file.
    
    Returns:
        dict: A dictionary containing the DOF information.
    """
    try:
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        
        # Calculate DOFs
        dof_info = {
            "NQ_POS = ": model.nq,  # Total position DOFs (qpos size)
            "NQ_VEL = ": model.nv,  # Total velocity DOFs (qvel size)
            "NV = ": model.njnt,  # Total number of joints (physical DOFs)
            "NU = ": model.nu,  # Number of actuators (controllable DOFs)
        }
        
        # Print DOF information
        print(f"DOF Information:")
        for key, value in dof_info.items():
            print(f"  {key}: {value}")
        
        return dof_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # with open("/home/rvsa/mj_playground_hand_env/para_env/xmls/reorient/reorient_hand.xml", "r") as f:
        # xml_file = f.read()
    with open("/home/rvsa/mj_playground_hand_env/para_env/xmls/rotateZ/rotateZ_hand_sim.xml", "r") as f:
        xml_file = f.read()
        
    calculate_dof(xml_file)