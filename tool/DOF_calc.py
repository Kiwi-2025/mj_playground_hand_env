# FILE: DOF_calc.py
import mujoco
import sys

def calculate_dof(xml_path):
    """
    Load a MuJoCo XML model and calculate its Degrees of Freedom (DOF).
    
    Args:
        xml_path (str): Path to the MuJoCo XML file.
    
    Returns:
        dict: A dictionary containing the DOF information.
    """
    try:
        # Load the MuJoCo model
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        # Calculate DOFs
        dof_info = {
            "NQ_POS": model.nq,  # Total position DOFs (qpos size)
            "NQ_VEL": model.nv,  # Total velocity DOFs (qvel size)
            "NV": model.njnt,  # Total number of joints (physical DOFs)
            "NU": model.nu,  # Number of actuators (controllable DOFs)
        }
        
        # Print DOF information
        print(f"DOF Information for {xml_path}:")
        for key, value in dof_info.items():
            print(f"  {key}: {value}")
        
        return dof_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Check if the XML file path is provided
    if len(sys.argv) < 2:
        print("Usage: python DOF_calc.py <path_to_xml>")
        sys.exit(1)
    
    # Get the XML file path from the command line
    xml_file = sys.argv[1]
    
    # Calculate DOF
    calculate_dof(xml_file)